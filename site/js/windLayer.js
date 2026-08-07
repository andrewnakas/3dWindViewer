// MapLibre custom layer: GPU wind particles at pressure-level altitudes.

import { QUAD_VERT, UPDATE_FRAG, DRAW_VERT, DRAW_FRAG } from "./shaders.js";
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
    out[info.name] = gl.getUniformLocation(prog, info.name);
  }
  return out;
}

class ParticleSystem {
  constructor(gl, level, size, useFloat) {
    this.level = level; // meta level entry
    this.size = size;   // state texture is size x size
    this.gl = gl;
    this.useFloat = useFloat;
    this.textures = [this.makeState(), this.makeState()];
    this.cur = 0;
    this.fbo = gl.createFramebuffer();
  }

  makeState() {
    const gl = this.gl;
    const n = this.size;
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    if (this.useFloat) {
      const data = new Float32Array(n * n * 4);
      for (let i = 0; i < n * n; i++) {
        data[i * 4] = Math.random();
        data[i * 4 + 1] = Math.random();
        data[i * 4 + 2] = Math.random() * MAX_AGE; // stagger ages
      }
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, n, n, 0, gl.RGBA, gl.FLOAT, data);
    } else {
      const data = new Uint8Array(n * n * 4);
      for (let i = 0; i < n * n; i++) {
        const x = Math.random(), y = Math.random();
        data[i * 4] = Math.floor((x * 255 % 1) * 256);      // fine
        data[i * 4 + 1] = Math.floor((y * 255 % 1) * 256);
        data[i * 4 + 2] = Math.floor(x * 255);              // coarse
        data[i * 4 + 3] = Math.floor(y * 255);
      }
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, n, n, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
    }
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return tex;
  }

  swap() { this.cur = 1 - this.cur; }
  get curTex() { return this.textures[this.cur]; }
  get prevTex() { return this.textures[1 - this.cur]; }

  destroy() {
    this.gl.deleteTexture(this.textures[0]);
    this.gl.deleteTexture(this.textures[1]);
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
    // Real atmospheric heights are invisible at continental scale (500 hPa is
    // ~5.5 km over a ~5000 km view), so particle altitude gets its own
    // exaggeration to make the level stack readable in 3D.
    this.altScale = opts.altScale ?? 15;
    this.opacity = 1.0;
    this.volumetric = opts.volumetric ?? true;
    this.levelIndices = this.volumetric
      ? volumetricIndices(meta)
      : [meta.levels.findIndex((l) => l.id === "10m")];
    this.particleCount = 65536;
    this.systems = [];
    this.frames = null;
    this.onReady = opts.onReady;
  }

  onAdd(map, gl) {
    if (typeof WebGL2RenderingContext === "undefined" || !(gl instanceof WebGL2RenderingContext)) {
      throw new Error("WebGL2 required");
    }
    // Prefer float state (adds per-particle age fade); fall back to the
    // universally supported RGBA8 packed encoding (iOS Safari etc.).
    this.useFloat = !!gl.getExtension("EXT_color_buffer_float")
      && !new URLSearchParams(location.search).has("bytestate");
    const defines = this.useFloat ? "#define FLOAT_STATE" : "";
    this.gl = gl;
    this.updateProg = compile(gl, QUAD_VERT, UPDATE_FRAG, defines);
    this.updateU = uniforms(gl, this.updateProg);
    this.drawProg = compile(gl, DRAW_VERT, DRAW_FRAG, defines);
    this.drawU = uniforms(gl, this.drawProg);
    this.vao = gl.createVertexArray(); // attribute-less draws still need a VAO

    this.rampTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.rampTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, rampTextureData());
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);

    this.frames = new FrameManager(gl, this.meta);
    this.rebuildSystems();
    this.onReady?.(this);
  }

  onRemove() {
    for (const s of this.systems) s.destroy();
    this.systems = [];
  }

  setLevels(indices) {
    this.levelIndices = indices;
    this.rebuildSystems();
  }

  setParticleCount(n) {
    this.particleCount = n;
    this.rebuildSystems();
  }

  rebuildSystems() {
    if (!this.gl) return;
    for (const s of this.systems) s.destroy();
    const perLevel = Math.max(1, Math.round(this.particleCount / this.levelIndices.length));
    const size = Math.max(16, 1 << Math.round(Math.log2(Math.sqrt(perLevel))));
    this.systems = this.levelIndices.map(
      (i) => new ParticleSystem(this.gl, this.meta.levels[i], size, this.useFloat)
    );
  }

  tileUniforms(level) {
    const { cols, rows } = this.meta.atlas;
    const col = level.index % cols;
    const row = Math.floor(level.index / cols);
    const sx = 1 / cols, sy = 1 / rows;
    // half-texel clamp (tile-local units) so LINEAR never bleeds across tiles
    const hx = 0.5 / (this.meta.tile.width * cols) / sx;
    const hy = 0.5 / (this.meta.tile.height * rows) / sy;
    return {
      offset: [col * sx, row * sy],
      scale: [sx, sy],
      clampMin: [hx, hy],
      clampMax: [1 - hx, 1 - hy],
    };
  }

  setTileUniforms(gl, U, sys) {
    const t = this.tileUniforms(sys.level);
    gl.uniform2fv(U.u_tileOffset, t.offset);
    gl.uniform2fv(U.u_tileScale, t.scale);
    gl.uniform2fv(U.u_clampMin, t.clampMin);
    gl.uniform2fv(U.u_clampMax, t.clampMax);
    gl.uniform2f(U.u_uRange, sys.level.uMin, sys.level.uMax);
    gl.uniform2f(U.u_vRange, sys.level.vMin, sys.level.vMax);
  }

  render(gl, matrix) {
    if (!this.systems.length || !this.frames) return;
    const pair = this.frames.getPair(this.time);
    this.frames.prefetch(this.time);
    if (!pair) { this.map.triggerRepaint(); return; }

    const b = this.meta.bounds;
    const lonSpan = b.east - b.west;
    const latSpan = b.north - b.south;

    // ---- update pass (offscreen) ----
    const prevFbo = gl.getParameter(gl.FRAMEBUFFER_BINDING);
    const prevVp = gl.getParameter(gl.VIEWPORT);

    gl.disable(gl.BLEND);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.STENCIL_TEST);
    gl.bindVertexArray(this.vao);
    gl.useProgram(this.updateProg);
    const U = this.updateU;
    gl.uniform1i(U.u_state, 0);
    gl.uniform1i(U.u_frameA, 1);
    gl.uniform1i(U.u_frameB, 2);
    gl.uniform1f(U.u_frameMix, pair.mix);
    gl.uniform1f(U.u_north, b.north);
    gl.uniform1f(U.u_lonSpan, lonSpan);
    gl.uniform1f(U.u_latSpan, latSpan);
    gl.uniform1f(U.u_dt, 90.0 * this.speedFactor); // simulated s per frame
    gl.uniform1f(U.u_maxAge, MAX_AGE);
    gl.uniform1f(U.u_time, (performance.now() % 100000) / 1000);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, pair.texA);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, pair.texB);

    for (const sys of this.systems) {
      this.setTileUniforms(gl, U, sys);
      gl.bindFramebuffer(gl.FRAMEBUFFER, sys.fbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, sys.prevTex, 0);
      gl.viewport(0, 0, sys.size, sys.size);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, sys.curTex);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      sys.swap();
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, prevFbo);
    gl.viewport(prevVp[0], prevVp[1], prevVp[2], prevVp[3]);

    // ---- draw pass ----
    const center = this.map.getCenter();
    const m2merc = maplibregl.MercatorCoordinate.fromLngLat(center, 1).z;

    gl.useProgram(this.drawProg);
    const D = this.drawU;
    gl.uniformMatrix4fv(D.u_matrix, false, matrix);
    gl.uniform1i(D.u_stateCurr, 0);
    gl.uniform1i(D.u_statePrev, 1);
    gl.uniform1i(D.u_ramp, 2);
    gl.uniform1i(D.u_frameA, 3);
    gl.uniform1i(D.u_frameB, 4);
    gl.uniform1f(D.u_frameMix, pair.mix);
    gl.uniform1f(D.u_west, b.west);
    gl.uniform1f(D.u_north, b.north);
    gl.uniform1f(D.u_lonSpan, lonSpan);
    gl.uniform1f(D.u_latSpan, latSpan);
    gl.uniform1f(D.u_streak, 4.0);
    gl.uniform1f(D.u_maxAge, MAX_AGE);
    gl.uniform1f(D.u_opacity, this.opacity);

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE); // additive glow on dark basemap
    gl.disable(gl.DEPTH_TEST); // low levels sit below terrain; keep visible

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.rampTex);
    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, pair.texA);
    gl.activeTexture(gl.TEXTURE4);
    gl.bindTexture(gl.TEXTURE_2D, pair.texB);

    for (const sys of this.systems) {
      this.setTileUniforms(gl, D, sys);
      gl.uniform1i(D.u_stateSize, sys.size);
      gl.uniform1f(D.u_alt, sys.level.heightMeters * m2merc * this.altScale);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, sys.curTex);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, sys.prevTex);
      gl.drawArrays(gl.LINES, 0, sys.size * sys.size * 2);
    }

    gl.bindVertexArray(null);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    this.map.triggerRepaint();
  }
}
