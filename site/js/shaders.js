// GLSL for the GPU particle system (WebGL2 / GLSL ES 3.00).
//
// One particle system spans a vertical stack of 1..MAX_STACK levels. Each
// particle carries (x, y) plus a continuous vertical coordinate sigma in
// [0,1] across the stack; the model's vertical velocity (omega, atlas B
// channel) moves particles between levels, so orographic uplift and
// convection are real 3D motion. A stack of 1 degenerates to classic
// single-level advection.
//
// State is two MRT targets (ping-ponged):
//   pos: FLOAT_STATE -> RG = xy;  byte mode -> webgl-wind packed hi/lo xy
//   aux: R = sigma, G = age/255 (float mode; byte mode uses stochastic drop)
//
// Atlas tile channels: R=u, G=v, B=omega, A=valid (in-domain AND above-ground).

export const MAX_STACK = 12;

export const QUAD_VERT = `#version 300 es
precision highp float;
out vec2 v_uv;
void main() {
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  v_uv = p;
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

const COMMON = `
#ifdef FLOAT_STATE
vec2 statePos(vec4 s) { return s.rg; }
#else
vec2 statePos(vec4 s) { return s.ba + s.rg / 255.0; }
#endif

float rand(vec2 co) {
  return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}

uniform sampler2D u_frameA;
uniform sampler2D u_frameB;
uniform float u_frameMix;
uniform int u_stackLen;
uniform vec2 u_tileOff[${MAX_STACK}];
uniform vec4 u_uvScale[${MAX_STACK}];   // uMin, uMax, vMin, vMax
uniform vec2 u_wScale[${MAX_STACK}];    // wMin, wMax (omega Pa/s)
uniform float u_wFactor[${MAX_STACK}];  // m/s per Pa/s at this level
uniform float u_height[${MAX_STACK}];   // meters ASL
uniform vec2 u_tileScale;
uniform vec2 u_clampMin;
uniform vec2 u_clampMax;

// Sampled wind at (pos, sigma): returns earth u, v (m/s), w (m/s), valid.
vec4 sampleWind(vec2 pos, float sigma, out float heightM) {
  float s = sigma * float(u_stackLen - 1);
  int j = u_stackLen > 1 ? int(clamp(s, 0.0, float(u_stackLen - 2))) : 0;
  int j1 = min(j + 1, u_stackLen - 1);
  float f = u_stackLen > 1 ? clamp(s - float(j), 0.0, 1.0) : 0.0;
  vec2 tl = clamp(pos, u_clampMin, u_clampMax) * u_tileScale;
  vec4 a = mix(texture(u_frameA, u_tileOff[j] + tl), texture(u_frameB, u_tileOff[j] + tl), u_frameMix);
  vec4 b = mix(texture(u_frameA, u_tileOff[j1] + tl), texture(u_frameB, u_tileOff[j1] + tl), u_frameMix);
  vec3 wa = vec3(mix(u_uvScale[j].x, u_uvScale[j].y, a.r),
                 mix(u_uvScale[j].z, u_uvScale[j].w, a.g),
                 mix(u_wScale[j].x, u_wScale[j].y, a.b) * u_wFactor[j]);
  vec3 wb = vec3(mix(u_uvScale[j1].x, u_uvScale[j1].y, b.r),
                 mix(u_uvScale[j1].z, u_uvScale[j1].w, b.g),
                 mix(u_wScale[j1].x, u_wScale[j1].y, b.b) * u_wFactor[j1]);
  heightM = mix(u_height[j], u_height[j1], f);
  // alpha blends toward the nearer level so particles pushed just above the
  // terrain (lower level masked below-ground) survive on the upper level's data
  return vec4(mix(wa, wb, f), mix(a.a, b.a, f));
}

float gapMeters(float sigma) {
  if (u_stackLen < 2) return 1.0;
  float s = sigma * float(u_stackLen - 1);
  int j = int(clamp(s, 0.0, float(u_stackLen - 2)));
  return max(u_height[j + 1] - u_height[j], 1.0);
}

uniform vec2 u_terrOff;    // atlas UV of the terrain tile
uniform vec2 u_terrRange;  // hMin, hMax meters

float terrainHeight(vec2 pos) {
  vec2 uv = u_terrOff + clamp(pos, u_clampMin, u_clampMax) * u_tileScale;
  vec4 t = texture(u_frameA, uv);  // terrain tile is identical in every frame
  return mix(u_terrRange.x, u_terrRange.y, (t.r * 255.0 * 256.0 + t.g * 255.0) / 65535.0);
}

// Lowest sigma whose level-stack height is >= h (piecewise-linear inverse).
float sigmaOfHeight(float h) {
  if (u_stackLen < 2 || h <= u_height[0]) return 0.0;
  for (int j = 0; j < ${MAX_STACK - 1}; j++) {
    if (j >= u_stackLen - 1) break;
    if (u_height[j + 1] >= h) {
      float f = clamp((h - u_height[j]) / max(u_height[j + 1] - u_height[j], 1.0), 0.0, 1.0);
      return (float(j) + f) / float(u_stackLen - 1);
    }
  }
  return 1.0;
}
`;

export const UPDATE_FRAG = `#version 300 es
precision highp float;

in vec2 v_uv;
layout(location = 0) out vec4 outPos;
layout(location = 1) out vec4 outAux;

uniform sampler2D u_statePos;
uniform sampler2D u_stateAux;
uniform float u_north;
uniform float u_lonSpan;
uniform float u_latSpan;
uniform float u_dt;
uniform float u_maxAge;
uniform float u_time;
uniform vec2 u_spawnMin;   // respawn region (normalized), follows the viewport
uniform vec2 u_spawnMax;

${COMMON}

void main() {
  vec4 sp = texture(u_statePos, v_uv);
  vec4 sa = texture(u_stateAux, v_uv);
  vec2 pos = statePos(sp);
  float sigma = sa.r;

  float heightM;
  vec4 wind = sampleWind(pos, sigma, heightM);

  float lat = u_north - pos.y * u_latSpan;
  float dlon = wind.x * u_dt / (111320.0 * max(cos(radians(lat)), 0.05));
  float dlat = wind.y * u_dt / 110540.0;
  vec2 npos = pos + vec2(dlon / u_lonSpan, -dlat / u_latSpan);
  float nsigma = sigma;
  if (u_stackLen > 1) {
    nsigma = clamp(sigma + (wind.z * u_dt) / gapMeters(sigma) / float(u_stackLen - 1), 0.0, 1.0);
    // terrain collision: flow is forced up and over rising ground
    nsigma = max(nsigma, sigmaOfHeight(terrainHeight(npos) + 5.0));
  }

  bool oob = npos.x < 0.0 || npos.x > 1.0 || npos.y < 0.0 || npos.y > 1.0 || wind.a < 0.5;
  vec2 seed = v_uv + fract(u_time);
  vec2 spawnPos = u_spawnMin + vec2(rand(seed), rand(seed.yx * 1.71)) * (u_spawnMax - u_spawnMin);
  float spawnSigma = rand(seed * 2.61);

#ifdef FLOAT_STATE
  float age = sa.g * 255.0 + 1.0;
  float lifetime = u_maxAge * (0.5 + rand(v_uv * 7.13));
  if (oob || age > lifetime) { npos = spawnPos; nsigma = spawnSigma; age = 0.0; }
  outPos = vec4(npos, 0.0, 1.0);
  outAux = vec4(nsigma, age / 255.0, 0.0, 1.0);
#else
  float speedN = length(wind.xy) / 60.0;
  float drop = step(1.0 - (0.006 + speedN * 0.012), rand(seed * 3.37));
  if (oob || drop > 0.5) { npos = spawnPos; nsigma = spawnSigma; }
  npos += (vec2(rand(seed * 5.1), rand(seed * 9.3)) - 0.5) / 65280.0;
  nsigma += (rand(seed * 6.7) - 0.5) / 255.0;  // dither vertical quantization
  outPos = vec4(fract(clamp(npos, 0.0, 1.0) * 255.0), floor(clamp(npos, 0.0, 1.0) * 255.0) / 255.0);
  outAux = vec4(clamp(nsigma, 0.0, 1.0), 0.0, 0.0, 1.0);
#endif
}`;

export const DRAW_VERT = `#version 300 es
precision highp float;

uniform sampler2D u_statePosCurr;
uniform sampler2D u_statePosPrev;
uniform sampler2D u_stateAuxCurr;
uniform int u_stateSize;
uniform mat4 u_matrix;
uniform float u_west;
uniform float u_north;
uniform float u_lonSpan;
uniform float u_latSpan;
uniform float u_altMerc;   // mercator z per meter of height-above-terrain (altitude scale)
uniform float u_exagMerc;  // mercator z per meter of terrain (matches map terrain exaggeration)
uniform float u_streak;
uniform float u_maxAge;

out float v_speed;
out float v_alpha;

const float PI = 3.141592653589793;

${COMMON}

void main() {
  int pid = gl_VertexID / 2;
  int end = gl_VertexID - pid * 2;
  ivec2 tc = ivec2(pid % u_stateSize, pid / u_stateSize);
  vec4 sc = texelFetch(u_statePosCurr, tc, 0);
  vec4 sp = texelFetch(u_statePosPrev, tc, 0);
  vec4 aux = texelFetch(u_stateAuxCurr, tc, 0);
  float sigma = aux.r;

  vec2 pc = statePos(sc);
  vec2 pp = statePos(sp);
  if (distance(pc, pp) > 0.02) pp = pc;
  vec2 pos = (end == 0) ? pc + (pp - pc) * u_streak : pc;

  float heightM;
  vec4 wind = sampleWind(pc, sigma, heightM);
  v_speed = clamp(length(wind.xy) / 60.0, 0.0, 1.0);

  float lon = u_west + pos.x * u_lonSpan;
  float lat = u_north - pos.y * u_latSpan;
  float mx = (lon + 180.0) / 360.0;
  float sm = clamp(sin(radians(lat)), -0.9999, 0.9999);
  float my = 0.5 - 0.25 * log((1.0 + sm) / (1.0 - sm)) / PI;

  // Terrain base rises with the map's own exaggeration so particles hug the
  // rendered surface; height above ground gets the (separate) altitude scale.
  float terr = terrainHeight(pc);
  float z = terr * u_exagMerc + max(heightM - terr, 8.0) * u_altMerc;
  gl_Position = u_matrix * vec4(mx, my, z, 1.0);

  float endDim = (end == 0) ? 0.2 : 1.0;
#ifdef FLOAT_STATE
  float age = aux.g * 255.0;
  float fadeIn = clamp(age / 10.0, 0.0, 1.0);
  float fadeOut = 1.0 - smoothstep(0.6, 1.0, age / u_maxAge);
  v_alpha = fadeIn * fadeOut * endDim;
#else
  v_alpha = 0.95 * endDim;
#endif
}`;

export const DRAW_FRAG = `#version 300 es
precision highp float;

in float v_speed;
in float v_alpha;
out vec4 outColor;

uniform sampler2D u_ramp;
uniform float u_opacity;

void main() {
  vec3 color = texture(u_ramp, vec2(v_speed, 0.5)).rgb;
  outColor = vec4(color, 1.0) * (v_alpha * u_opacity);
}`;
