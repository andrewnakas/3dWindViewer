// GLSL for the GPU particle system (WebGL2 / GLSL ES 3.00).
//
// Two state-texture modes, chosen at runtime:
//  - FLOAT_STATE: RGBA32F state (RG = pos, B = age). Needs EXT_color_buffer_float.
//  - byte mode:   RGBA8 state, position packed hi/lo per axis (webgl-wind style:
//                 rg = fine, ba = coarse). Works on every WebGL2 device (iOS).
//                 No age channel -> stochastic respawn.
// Particle speed/color is sampled from the wind atlas in the draw shader in
// both modes.

export const QUAD_VERT = `#version 300 es
precision highp float;
out vec2 v_uv;
void main() {
  // fullscreen triangle
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  v_uv = p;
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

const STATE_HELPERS = `
#ifdef FLOAT_STATE
vec2 statePos(vec4 s) { return s.rg; }
#else
vec2 statePos(vec4 s) { return s.ba + s.rg / 255.0; }
vec4 encodeState(vec2 p) { return vec4(fract(p * 255.0), floor(p * 255.0) / 255.0); }
#endif

float rand(vec2 co) {
  return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}
`;

export const UPDATE_FRAG = `#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 outState;

uniform sampler2D u_state;
uniform sampler2D u_frameA;
uniform sampler2D u_frameB;
uniform float u_frameMix;

uniform vec2 u_tileOffset;   // atlas UV of tile origin
uniform vec2 u_tileScale;    // tile extent in atlas UV
uniform vec2 u_clampMin;     // half-texel clamp inside tile (tile-local 0..1)
uniform vec2 u_clampMax;
uniform vec2 u_uRange;       // uMin, uMax (m/s)
uniform vec2 u_vRange;
uniform float u_north;
uniform float u_lonSpan;
uniform float u_latSpan;
uniform float u_dt;          // simulated seconds per rendered frame
uniform float u_maxAge;
uniform float u_time;        // random seed

${STATE_HELPERS}

void main() {
  vec4 st = texture(u_state, v_uv);
  vec2 pos = statePos(st);

  vec2 tuv = u_tileOffset + clamp(pos, u_clampMin, u_clampMax) * u_tileScale;
  vec4 w = mix(texture(u_frameA, tuv), texture(u_frameB, tuv), u_frameMix);
  float u = mix(u_uRange.x, u_uRange.y, w.r);
  float v = mix(u_vRange.x, u_vRange.y, w.g);
  float speedN = length(vec2(u, v)) / 60.0;

  float lat = u_north - pos.y * u_latSpan;
  float dlon = u * u_dt / (111320.0 * max(cos(radians(lat)), 0.05));
  float dlat = v * u_dt / 110540.0;
  vec2 npos = pos + vec2(dlon / u_lonSpan, -dlat / u_latSpan);

  bool oob = npos.x < 0.0 || npos.x > 1.0 || npos.y < 0.0 || npos.y > 1.0 || w.a < 0.5;
  vec2 seed = v_uv + fract(u_time);
  vec2 spawn = vec2(rand(seed), rand(seed.yx * 1.71));

#ifdef FLOAT_STATE
  float age = st.b + 1.0;
  float lifetime = u_maxAge * (0.5 + rand(v_uv * 7.13));
  if (oob || age > lifetime) { npos = spawn; age = 0.0; }
  outState = vec4(npos, age, 0.0);
#else
  // stochastic respawn: base rate + extra for fast particles (keeps density even)
  float drop = step(1.0 - (0.006 + speedN * 0.012), rand(seed * 3.37));
  if (oob || drop > 0.5) npos = spawn;
  // dither below the 16-bit quantum so slow particles don't freeze
  npos += (vec2(rand(seed * 5.1), rand(seed * 9.3)) - 0.5) / 65280.0;
  outState = encodeState(clamp(npos, 0.0, 1.0));
#endif
}`;

// Draw particles as short streak segments (GL_LINES, 2 verts per particle).
export const DRAW_VERT = `#version 300 es
precision highp float;

uniform sampler2D u_stateCurr;
uniform sampler2D u_statePrev;
uniform sampler2D u_frameA;
uniform sampler2D u_frameB;
uniform float u_frameMix;
uniform vec2 u_tileOffset;
uniform vec2 u_tileScale;
uniform vec2 u_clampMin;
uniform vec2 u_clampMax;
uniform vec2 u_uRange;
uniform vec2 u_vRange;
uniform int u_stateSize;
uniform mat4 u_matrix;
uniform float u_west;
uniform float u_north;
uniform float u_lonSpan;
uniform float u_latSpan;
uniform float u_alt;       // altitude in mercator z units (premultiplied)
uniform float u_streak;    // tail extension factor
uniform float u_maxAge;

out float v_speed;
out float v_alpha;

const float PI = 3.141592653589793;

${STATE_HELPERS}

void main() {
  int pid = gl_VertexID / 2;
  int end = gl_VertexID - pid * 2;
  ivec2 tc = ivec2(pid % u_stateSize, pid / u_stateSize);
  vec4 sc = texelFetch(u_stateCurr, tc, 0);
  vec4 sp = texelFetch(u_statePrev, tc, 0);

  vec2 pc = statePos(sc);
  vec2 pp = statePos(sp);
  if (distance(pc, pp) > 0.02) pp = pc;  // collapse respawn jumps
  vec2 pos = (end == 0) ? pc + (pp - pc) * u_streak : pc;

  vec2 tuv = u_tileOffset + clamp(pc, u_clampMin, u_clampMax) * u_tileScale;
  vec4 w = mix(texture(u_frameA, tuv), texture(u_frameB, tuv), u_frameMix);
  float wu = mix(u_uRange.x, u_uRange.y, w.r);
  float wv = mix(u_vRange.x, u_vRange.y, w.g);
  v_speed = clamp(length(vec2(wu, wv)) / 60.0, 0.0, 1.0);

  float lon = u_west + pos.x * u_lonSpan;
  float lat = u_north - pos.y * u_latSpan;
  float mx = (lon + 180.0) / 360.0;
  float s = clamp(sin(radians(lat)), -0.9999, 0.9999);
  float my = 0.5 - 0.25 * log((1.0 + s) / (1.0 - s)) / PI;

  gl_Position = u_matrix * vec4(mx, my, u_alt, 1.0);

  float endDim = (end == 0) ? 0.1 : 1.0;
#ifdef FLOAT_STATE
  float fadeIn = clamp(sc.b / 10.0, 0.0, 1.0);
  float fadeOut = 1.0 - smoothstep(0.6, 1.0, sc.b / u_maxAge);
  v_alpha = fadeIn * fadeOut * endDim;
#else
  v_alpha = 0.85 * endDim;
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
