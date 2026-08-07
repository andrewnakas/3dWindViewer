// MapLibre custom layer: GPU wind particles with a continuous vertical
// coordinate — one system spans the whole level stack and particles rise and
// sink with the model's vertical velocity (see shaders.js).

import { QUAD_VERT, UPDATE_FRAG, DRAW_VERT, DRAW_FRAG, MAX_STACK } from "./shaders.js";
import { rampTextureData, volumetricIndices } from "./atmosphere.js";
import { FrameManager } from "./frames.js";

const MAX_AGE = 90; // frames (float-state mode only)

function compile(gl, vertSrc, fragSrc, defines = "") {
  const inject = (src) => src.replace("#version 300 es", `#version 300 es\n${defines}`);
  const prog = gl.createProgram();
  for (const [type, src] of [[gl.VERTEX_SHADER, inject(vertSrc)], [gl.FRAGMENT_SHADER, inject(fragSrc)]]) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      throw new Error("shader: " + gl.getShaderInfoLog(sh));
    }
    gl.attachShader(prog, sh);
  }
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    throw new Error("link: " + gl.getProgramInfoLog(prog));
  }
  return prog;
}

function uniforms(gl, prog) {
  const out = {};
  const n = gl.getProgramParameter(prog, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < n; i++) {
    const info = gl.getActiveUniform(prog, i);
    out[info.name.replace("[0]", "")] = gl.getUniformLocation(prog, info.name);
  }
  return out;
}

class ParticleSystem {
  constructor(gl, levels, size, useFloat) {
    this.levels = levels; // meta level entries, surface -> top
    this.size = size;
    this.gl = gl;
    this.useFloat = useFloat;
    // ping-pong pairs of MRT state: [ {pos, aux}, {pos, aux} ]
    this.state = [this.makeState(), this.makeState()];
    this.cur = 0;
    this.fbo = gl.createFramebuffer();
  }

  makeState() {
    const gl = this.gl;
    const n = this.size;
    const make = (fill) => {
      const tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      if (this.useFloat) {
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, n, n, 0, gl.RGBA, gl.FLOAT, fill.f32);
      } else {
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, n, n, 0, gl.RGBA, gl.UNSIGNED_BYTE, fill.u8);
      }
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.bindTexture(gl.TEXTURE_2D, null);
      return tex;
    };

    const count = n * n;
    const posF = new Float32Array(count * 4);
    const posB = new Uint8Array(count * 4);
    const auxF = new Float32Array(count * 4);
    const auxB = new Uint8Array(count * 4);
    for (let i = 0; i < count; i++) {
      const x = Math.random(), y = Math.random(), sigma = Math.random();
      posF[i * 4] = x; posF[i * 4 + 1] = y;
      posB[i * 4] = Math.floor((x * 255 % 1) * 256);
      posB[i * 4 + 1] = Math.floor((y * 255 % 1) * 256);
      posB[i * 4 + 2] = Math.floor(x * 255);
      posB[i * 4 + 3] = Math.floor(y * 255);
      auxF[i * 4] = sigma; auxF[i * 4 + 1] = (Math.random() * MAX_AGE) / 255;
      auxB[i * 4] = Math.floor(sigma * 255);
    }
    return {
      pos: make({ f32: posF, u8: posB }),
      aux: make({ f32: auxF, u8: auxB }),
    };
  }

  swap() { this.cur = 1 - this.cur; }
  get curState() { return this.state[this.cur]; }
  get prevState() { return this.state[1 - this.cur]; }

  destroy() {
    for (const s of this.state) {
      this.gl.deleteTexture(s.pos);
      this.gl.deleteTexture(s.aux);
    }
    this.gl.deleteFramebuffer(this.fbo);
  }
}

export class WindLayer {
  constructor(map, meta, opts = {}) {
    this.id = "wind-particles";
    this.type = "custom";
    this.renderingMode = "3d";
    this.map = map;
    this.meta = meta;
    this.time = 0;              // forecast hour, continuous
    this.speedFactor = 1.0;
    this.exaggeration = opts.exaggeration ?? 1.5;
    // Real atmospheric heights are invisible at continental scale, so particle
    // altitude gets its own exaggeration to make the stack readable in 3D.
    this.altScale = opts.altScale ?? 15;
    this.opacity = 1.0;
    this.volumetric = opts.volumetric ?? true;
    this.levelIndices = this.volumetric
      ? volumetricIndices(meta)
      : [meta.levels.findIndex((l) => l.id === "10m")];
    this.particleCount = 65536;
    this.system = null;
    this.frames = null;
    this.onReady = opts.onReady;
  }

  onAdd(map, gl) {
    if (typeof WebGL2RenderingContext === "undefined" || !(gl instanceof WebGL2RenderingContext)) {
      throw new Error("WebGL2 required");
    }
    this.useFloat = !!gl.getExtension("EXT_color_buffer_float")
      && !new URLSearchParams(location.search).has("bytestate");
    const defines = this.useFloat ? "#define FLOAT_STATE" : "";
    this.gl = gl;
    this.updateProg = compile(gl, QUAD_VERT, UPDATE_FRAG, defines);
    this.updateU = uniforms(gl, this.updateProg);
    this.drawProg = compile(gl, DRAW_VERT, DRAW_FRAG, defines);
    this.drawU = uniforms(gl, this.drawProg);
    this.vao = gl.createVertexArray();

    this.rampTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.rampTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, rampTextureData());
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);

    this.frames = new FrameManager(gl, this.meta);
    this.rebuildSystem();
    this.onReady?.(this);
  }

  onRemove() {
    this.system?.destroy();
    this.system = null;
  }

  setLevels(indices) {
    this.levelIndices = indices.slice(0, MAX_STACK);
    this.rebuildSystem();
  }

  setParticleCount(n) {
    this.particleCount = n;
    this.rebuildSystem();
  }

  rebuildSystem() {
    if (!this.gl) return;
    this.system?.destroy();
    const size = Math.max(16, 1 << Math.floor(Math.log2(Math.sqrt(this.particleCount))));
    const levels = this.levelIndices
      .map((i) => this.meta.levels[i])
      .sort((a, b) => a.heightMeters - b.heightMeters);
    this.system = new ParticleSystem(this.gl, levels, size, this.useFloat);
    this.stackUniforms = this.buildStackUniforms(levels);
  }

  buildStackUniforms(levels) {
    const { cols, rows } = this.meta.atlas;
    const L = levels.length;
    const tileOff = new Float32Array(MAX_STACK * 2);
    const uvScale = new Float32Array(MAX_STACK * 4);
    const wScale = new Float32Array(MAX_STACK * 2);
    const wFactor = new Float32Array(MAX_STACK);
    const height = new Float32Array(MAX_STACK);
    for (let k = 0; k < MAX_STACK; k++) {
      const lv = levels[Math.min(k, L - 1)];
      tileOff[k * 2] = (lv.index % cols) / cols;
      tileOff[k * 2 + 1] = Math.floor(lv.index / cols) / rows;
      uvScale.set([lv.uMin, lv.uMax, lv.vMin, lv.vMax], k * 4);
      wScale.set([lv.wMin ?? -1, lv.wMax ?? 1], k * 2);
      wFactor[k] = lv.wFactor ?? 0;
      height[k] = lv.heightMeters;
    }
    const hx = 0.5 / (this.meta.tile.width * cols) * cols;
    const hy = 0.5 / (this.meta.tile.height * rows) * rows;
    return {
      len: L, tileOff, uvScale, wScale, wFactor, height,
      tileScale: [1 / cols, 1 / rows],
      clampMin: [hx, hy],
      clampMax: [1 - hx, 1 - hy],
    };
  }

  setStackUniforms(gl, U) {
    const s = this.stackUniforms;
    gl.uniform1i(U.u_stackLen, s.len);
    gl.uniform2fv(U.u_tileOff, s.tileOff);
    gl.uniform4fv(U.u_uvScale, s.uvScale);
    gl.uniform2fv(U.u_wScale, s.wScale);
    gl.uniform1fv(U.u_wFactor, s.wFactor);
    gl.uniform1fv(U.u_height, s.height);
    gl.uniform2fv(U.u_tileScale, s.tileScale);
    gl.uniform2fv(U.u_clampMin, s.clampMin);
    gl.uniform2fv(U.u_clampMax, s.clampMax);
  }

  render(gl, matrix) {
    const sys = this.system;
    if (!sys || !this.frames) return;
    const pair = this.frames.getPair(this.time);
    this.frames.prefetch(this.time);
    if (!pair) { this.map.triggerRepaint(); return; }

    const b = this.meta.bounds;
    const lonSpan = b.east - b.west;
    const latSpan = b.north - b.south;

    // ---- update pass (offscreen, MRT) ----
    const prevFbo = gl.getParameter(gl.FRAMEBUFFER_BINDING);
    const prevVp = gl.getParameter(gl.VIEWPORT);

    gl.disable(gl.BLEND);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.STENCIL_TEST);
    gl.bindVertexArray(this.vao);
    gl.useProgram(this.updateProg);
    const U = this.updateU;
    this.setStackUniforms(gl, U);
    gl.uniform1i(U.u_statePos, 0);
    gl.uniform1i(U.u_stateAux, 1);
    gl.uniform1i(U.u_frameA, 2);
    gl.uniform1i(U.u_frameB, 3);
    gl.uniform1f(U.u_frameMix, pair.mix);
    gl.uniform1f(U.u_north, b.north);
    gl.uniform1f(U.u_lonSpan, lonSpan);
    gl.uniform1f(U.u_latSpan, latSpan);
    gl.uniform1f(U.u_dt, 90.0 * this.speedFactor);
    gl.uniform1f(U.u_maxAge, MAX_AGE);
    gl.uniform1f(U.u_time, (performance.now() % 100000) / 1000);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, sys.curState.pos);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, sys.curState.aux);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, pair.texA);
    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, pair.texB);

    gl.bindFramebuffer(gl.FRAMEBUFFER, sys.fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, sys.prevState.pos, 0);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, sys.prevState.aux, 0);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
    gl.viewport(0, 0, sys.size, sys.size);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    sys.swap();

    gl.bindFramebuffer(gl.FRAMEBUFFER, prevFbo);
    gl.viewport(prevVp[0], prevVp[1], prevVp[2], prevVp[3]);

    // ---- draw pass ----
    const center = this.map.getCenter();
    const m2merc = maplibregl.MercatorCoordinate.fromLngLat(center, 1).z;

    gl.useProgram(this.drawProg);
    const D = this.drawU;
    this.setStackUniforms(gl, D);
    gl.uniformMatrix4fv(D.u_matrix, false, matrix);
    gl.uniform1i(D.u_statePosCurr, 0);
    gl.uniform1i(D.u_statePosPrev, 1);
    gl.uniform1i(D.u_stateAuxCurr, 2);
    gl.uniform1i(D.u_ramp, 4);
    gl.uniform1i(D.u_frameA, 5);
    gl.uniform1i(D.u_frameB, 6);
    gl.uniform1f(D.u_frameMix, pair.mix);
    gl.uniform1f(D.u_west, b.west);
    gl.uniform1f(D.u_north, b.north);
    gl.uniform1f(D.u_lonSpan, lonSpan);
    gl.uniform1f(D.u_latSpan, latSpan);
    gl.uniform1f(D.u_altMerc, m2merc * this.altScale);
    gl.uniform1f(D.u_streak, 4.0);
    gl.uniform1f(D.u_maxAge, MAX_AGE);
    gl.uniform1f(D.u_opacity, this.opacity);
    gl.uniform1i(D.u_stateSize, sys.size);

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE); // additive glow on dark basemap
    gl.disable(gl.DEPTH_TEST);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, sys.curState.pos);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, sys.prevState.pos);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, sys.curState.aux);
    gl.activeTexture(gl.TEXTURE4);
    gl.bindTexture(gl.TEXTURE_2D, this.rampTex);
    gl.activeTexture(gl.TEXTURE5);
    gl.bindTexture(gl.TEXTURE_2D, pair.texA);
    gl.activeTexture(gl.TEXTURE6);
    gl.bindTexture(gl.TEXTURE_2D, pair.texB);

    gl.drawArrays(gl.LINES, 0, sys.size * sys.size * 2);

    gl.bindVertexArray(null);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    this.map.triggerRepaint();
  }
}
