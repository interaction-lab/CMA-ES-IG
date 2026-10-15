// Interactive CMA-ES-IG demo: rank Jaco2 handovers, entirely in the browser.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";
import URDFLoader from "urdf-loader";
import { ALGORITHMS, ParticleBelief, RNG, eighSym, standardise } from "./cmaesig.js";

const ROOT = new URL("..", import.meta.url).href; // .../static/
const N_TRAJ = 1000, T = 50, DOF = 6, FEAT = 8;
const K = 3;
const LANE_SPACING = 1.4;
const LANES = [
  { name: "A", color: "#0072b2" },
  { name: "B", color: "#d55e00" },
  { name: "C", color: "#009e73" },
];
const BEST_COLOR = "#e69f00";
const REAL_DURATION = 9; // seconds per handover on the real robot
const HOLD = 1.2;
const ARM_JOINTS = [1, 2, 3, 4, 5, 6].map((i) => `j2n6s300_joint_${i}`);
// Finger angles and cup offset chosen so the fingertips touch the cup without intersecting it.
const GRIP = { finger: 0.7, finger_tip: 0.1 };
const EE_LINK = "j2n6s300_end_effector";
const PLACES = ["1st", "2nd", "3rd"];
const CAPTIONS = {
  "CMA-ES-IG": 'CMA-ES-IG draws 100 samples (<i class="sw sw-sample"></i>) from its search distribution (<i class="sw sw-dist"></i>), clusters them with K-means, and shows you the handovers nearest the 3 centroids.',
  "CMA-ES": 'CMA-ES samples the 3 candidates directly from its search distribution (<i class="sw sw-dist"></i>), so they are often close together and harder to tell apart.',
  Infogain: "Infogain picks the 3 handovers whose ranking would tell the robot the most about your preference. It has no search distribution, so it doesn't steer toward what you like.",
};

const $ = (sel, el = document) => el.querySelector(sel);

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------
async function loadF32(path) {
  const res = await fetch(ROOT + path);
  if (!res.ok) throw new Error(`Could not load ${path}`);
  return new Float32Array(await res.arrayBuffer());
}

function pca2(rows) {
  const d = rows[0].length, n = rows.length;
  const mean = new Array(d).fill(0);
  for (const r of rows) for (let j = 0; j < d; j++) mean[j] += r[j] / n;
  const cov = Array.from({ length: d }, () => new Array(d).fill(0));
  for (const r of rows) for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) cov[i][j] += ((r[i] - mean[i]) * (r[j] - mean[j])) / n;
  const { values, vectors } = eighSym(cov);
  const order = values.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).map(([, i]) => i);
  const P = order.slice(0, 2).map((c) => vectors.map((row) => row[c]));
  const project = (x) => P.map((p) => p.reduce((s, v, j) => s + v * (x[j] - mean[j]), 0));
  return { P, mean, project };
}

// Catmull-Rom interpolation of a (T x DOF) trajectory at phase in [0, 1].
function sampleTraj(traj, phase, out) {
  const u = Math.min(Math.max(phase, 0), 1) * (T - 1);
  const i = Math.min(Math.floor(u), T - 2), t = u - i;
  const at = (k, j) => traj[Math.min(Math.max(k, 0), T - 1) * DOF + j];
  for (let j = 0; j < DOF; j++) {
    const p0 = at(i - 1, j), p1 = at(i, j), p2 = at(i + 1, j), p3 = at(i + 2, j);
    out[j] = 0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------
function makeMaterials() {
  return {
    carbon: new THREE.MeshStandardMaterial({ color: 0x2a2c30, roughness: 0.45, metalness: 0.25 }),
    ring: new THREE.MeshStandardMaterial({ color: 0xb9bcc2, roughness: 0.3, metalness: 0.8 }),
    finger: new THREE.MeshStandardMaterial({ color: 0x3c3f45, roughness: 0.6, metalness: 0.1 }),
  };
}

function loadRobot(materials) {
  return new Promise((resolve, reject) => {
    const manager = new THREE.LoadingManager();
    const loader = new URDFLoader(manager);
    loader.loadMeshCb = (path, mgr, done) => {
      loader.defaultMeshLoader(path, mgr, (obj, err) => {
        if (obj) {
          const mat = /ring/.test(path) ? materials.ring : /finger/.test(path) ? materials.finger : materials.carbon;
          obj.traverse((c) => {
            if (c.isMesh) { c.material = mat; c.castShadow = true; c.receiveShadow = true; }
          });
        }
        done(obj, err);
      });
    };
    let robot;
    manager.onLoad = () => resolve(robot);
    manager.onError = (url) => reject(new Error(`Failed to load ${url}`));
    loader.load(ROOT + "robot/j2n6s300/j2n6s300.urdf", (r) => { robot = r; }, undefined, reject);
  });
}

function setArm(robot, q) {
  for (let j = 0; j < DOF; j++) robot.setJointValue(ARM_JOINTS[j], q[j]);
}

function closeGripper(robot) {
  for (let f = 1; f <= 3; f++) {
    robot.setJointValue(`j2n6s300_joint_finger_${f}`, GRIP.finger);
    robot.setJointValue(`j2n6s300_joint_finger_tip_${f}`, GRIP.finger_tip);
  }
}

function makeCup() {
  const g = new THREE.Group();
  const mat = new THREE.MeshStandardMaterial({ color: 0x2f6fd0, roughness: 0.45, side: THREE.DoubleSide });
  const body = new THREE.Mesh(new THREE.CylinderGeometry(0.036, 0.03, 0.1, 32, 1, true), mat);
  const bottom = new THREE.Mesh(new THREE.CircleGeometry(0.03, 32), mat);
  bottom.rotation.x = Math.PI / 2;
  bottom.position.y = -0.05;
  for (const m of [body, bottom]) { m.castShadow = true; g.add(m); }
  // three.js cylinders run along +Y, across the gripper; +Z is the approach direction
  // and the palm ends at z = -0.039, so the cup sits just in front of it.
  g.position.set(0, 0, 0.02);
  return g;
}

function makeTable(group) {
  const top = new THREE.MeshStandardMaterial({ color: 0x8a6d55, roughness: 0.8 });
  const leg = new THREE.MeshStandardMaterial({ color: 0x5c4838, roughness: 0.8 });
  const table = new THREE.Group();
  const t = new THREE.Mesh(new THREE.BoxGeometry(0.95, 0.9, 0.04), top);
  t.position.set(0.3, 0.05, -0.02);
  t.receiveShadow = true;
  table.add(t);
  for (const dx of [-0.12, 0.72]) for (const dy of [-0.34, 0.44]) {
    const l = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.04, 0.7), leg);
    l.position.set(dx, dy, -0.39);
    l.castShadow = true;
    table.add(l);
  }
  group.add(table);
}

// Minimum-jerk time scaling: the robot starts and stops at rest (zero velocity
// and acceleration) while following the same path.
const retime = (t) => t * t * t * (10 - 15 * t + 6 * t * t);

// A translucent seated person. They stay put while the robot rig is rotated.
// Seated where the (rotated, shifted) handovers end, around x = 0.67, y = -0.52, z = 0.30.
const PERSON = { x: 1.02, y: -0.52 };
const RIG_OFFSET = 0.35; // robot + table slide this far toward the person
function makePerson(group) {
  const ghost = new THREE.MeshStandardMaterial({ color: 0x7d8fa8, roughness: 0.7, transparent: true, opacity: 0.5, depthWrite: false });
  const chairMat = new THREE.MeshStandardMaterial({ color: 0x5a5f66, roughness: 0.8, transparent: true, opacity: 0.45, depthWrite: false });
  const person = new THREE.Group();
  person.position.set(PERSON.x, PERSON.y, 0);
  const add = (geom, mat, x, y, z) => { const m = new THREE.Mesh(geom, mat); m.position.set(x, y, z); person.add(m); return m; };
  // limb between two points (in the person's frame)
  const limb = (a, b, r) => {
    const A = new THREE.Vector3(...a), B = new THREE.Vector3(...b);
    const m = add(new THREE.CapsuleGeometry(r, A.distanceTo(B), 6, 12), ghost, ...A.clone().add(B).multiplyScalar(0.5).toArray());
    m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), B.clone().sub(A).normalize());
  };
  // chair
  add(new THREE.BoxGeometry(0.42, 0.42, 0.04), chairMat, 0.05, 0, -0.31);
  add(new THREE.BoxGeometry(0.04, 0.42, 0.45), chairMat, 0.25, 0, -0.08);
  for (const dx of [-0.13, 0.23]) for (const dy of [-0.18, 0.18]) add(new THREE.BoxGeometry(0.03, 0.03, 0.41), chairMat, dx, dy, -0.535);
  // body, facing the robot (-x)
  limb([0.06, 0, -0.2], [0.04, 0, 0.12], 0.13);                     // torso
  add(new THREE.SphereGeometry(0.095, 24, 16), ghost, 0.02, 0, 0.36); // head
  for (const dy of [-0.09, 0.09]) {
    limb([0.0, dy, -0.25], [-0.32, dy, -0.25], 0.055);              // thighs
    limb([-0.34, dy, -0.27], [-0.36, dy, -0.7], 0.045);             // shins
  }
  for (const side of [-1, 1]) limb([0.02, 0.17 * side, 0.12], [0.08, 0.21 * side, -0.12], 0.04); // arms at sides
  person.traverse((c) => { c.renderOrder = 1; });
  const el = document.createElement("div");
  el.className = "you-label";
  el.textContent = "You";
  const label = new CSS2DObject(el);
  label.position.set(0.02, 0, 0.55);
  person.add(label);
  group.add(person);
  return person;
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------
export async function initDemo(container) {
  const viewport = $(".demo-viewport", container);
  const status = $(".demo-loading", container);
  const setStatus = (msg) => { status.textContent = msg; };

  setStatus("Loading handovers…");
  const [trajRaw, embRaw] = await Promise.all([loadF32("data/handovers.f32"), loadF32("data/embeddings.f32")]);
  const trajectories = Array.from({ length: N_TRAJ }, (_, i) => trajRaw.subarray(i * T * DOF, (i + 1) * T * DOF));
  const features = standardise(Array.from({ length: N_TRAJ }, (_, i) => Array.from(embRaw.subarray(i * FEAT, (i + 1) * FEAT))));
  const pca = pca2(features);
  const features2d = features.map(pca.project);

  // --- renderer / scene -----------------------------------------------------
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  viewport.appendChild(renderer.domElement);
  const labelRenderer = new CSS2DRenderer();
  labelRenderer.domElement.className = "demo-labels";
  viewport.appendChild(labelRenderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf3f1ec);
  scene.fog = new THREE.Fog(0xf3f1ec, 6, 14);
  const camera = new THREE.PerspectiveCamera(40, 1, 0.05, 50);
  camera.up.set(0, 0, 1);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.maxPolarAngle = Math.PI * 0.49;
  controls.minDistance = 0.6;
  controls.maxDistance = 7;

  scene.add(new THREE.HemisphereLight(0xffffff, 0xb9ab98, 1.6));
  const sun = new THREE.DirectionalLight(0xffffff, 2.2);
  sun.position.set(2.5, -1.5, 4);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, { left: -3, right: 3, top: 3, bottom: -3, near: 0.5, far: 12 });
  sun.shadow.bias = -0.0005;
  scene.add(sun);
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(30, 30), new THREE.MeshStandardMaterial({ color: 0xe9e5dd, roughness: 1 }));
  floor.position.z = -0.74;
  floor.receiveShadow = true;
  scene.add(floor);
  const grid = new THREE.GridHelper(12, 48, 0xd6d0c4, 0xdfdad0);
  grid.rotation.x = Math.PI / 2;
  grid.position.z = -0.739;
  scene.add(grid);

  setStatus("Loading the Jaco2 model…");
  const materials = makeMaterials();
  const baseRobot = await loadRobot(materials);
  const helper = baseRobot.clone(); // used for forward kinematics only
  closeGripper(helper);

  const lanes = LANES.map((spec, i) => {
    const group = new THREE.Group();
    group.position.y = (i - 1) * LANE_SPACING;
    scene.add(group);
    // Robot, table and hand paths form a rig turned 90° clockwise (seen from above)
    // about the robot base; the person stays where they are.
    const rig = new THREE.Group();
    rig.rotation.z = -Math.PI / 2;
    rig.position.x = RIG_OFFSET;
    group.add(rig);
    makeTable(rig);
    makePerson(group);
    const plate = new THREE.Mesh(new THREE.CylinderGeometry(0.11, 0.11, 0.012, 40), new THREE.MeshStandardMaterial({ color: spec.color, roughness: 0.6 }));
    plate.rotation.x = Math.PI / 2;
    plate.position.z = 0.006;
    rig.add(plate);
    const robot = baseRobot.clone();
    closeGripper(robot);
    robot.links[EE_LINK].add(makeCup());
    rig.add(robot);
    robot.traverse((c) => { if (c.isMesh) c.userData.lane = i; });
    const el = document.createElement("div");
    el.className = "lane-label";
    el.style.setProperty("--lane", spec.color);
    const label = new CSS2DObject(el);
    label.position.set(RIG_OFFSET, 0, 0.82);
    group.add(label);
    return { ...spec, i, group, rig, robot, plate, el, trail: null, index: -1 };
  });

  // --- state ------------------------------------------------------------------
  const ui = {
    round: $("[data-round]", container),
    cards: [...container.querySelectorAll("[data-card]")],
    submit: $("[data-submit]", container),
    clear: $("[data-clear]", container),
    best: $("[data-best]", container),
    hint: $("[data-hint]", container),
    algos: [...container.querySelectorAll("[data-algo]")],
    caption: $("[data-caption]", container),
    restart: $("[data-restart]", container),
    speed: $("[data-speed]", container),
    paths: $("[data-paths]", container),
    replay: $("[data-replay]", container),
    views: [...container.querySelectorAll("[data-view]")],
    space: $("[data-space]", container),
    thinking: $("[data-thinking]", container),
  };
  const state = {
    method: "CMA-ES-IG",
    rng: null, belief: null, algorithm: null,
    query: [], order: [], history: [], round: 0,
    speed: parseFloat(ui.speed.value), restartAt: performance.now(), mode: "rank", busy: false,
  };

  function reset() {
    const seed = (Math.random() * 2 ** 32) >>> 0;
    state.rng = new RNG(seed);
    state.belief = new ParticleBelief(FEAT, new RNG(seed ^ 0x9e3779b9));
    state.algorithm = new ALGORITHMS[state.method](features, { k: K, rng: new RNG(seed ^ 0x85ebca6b) });
    state.round = 0;
    state.history = [];
    state.order = [];
    state.mode = "rank";
    state.query = state.algorithm.query(state.belief);
    showQuery();
  }

  const bestIndex = () => {
    const w = state.belief.mean();
    let best = 0, bv = -Infinity;
    features.forEach((f, i) => { const v = f.reduce((s, x, j) => s + x * w[j], 0); if (v > bv) { bv = v; best = i; } });
    return best;
  };

  // --- trajectories in the scene --------------------------------------------
  const q = new Float64Array(DOF);
  const tmp = new THREE.Vector3();

  function setTrajectory(lane, index, color) {
    lane.index = index;
    const traj = trajectories[index];
    const pts = [];
    for (let s = 0; s <= 60; s++) {
      setArm(helper, sampleTraj(traj, s / 60, q));
      helper.updateMatrixWorld(true);
      helper.links[EE_LINK].getWorldPosition(tmp);
      pts.push(tmp.clone());
    }
    if (lane.trail) { lane.rig.remove(lane.trail); lane.trail.geometry.dispose(); }
    lane.trail = new THREE.Mesh(
      new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 120, 0.006, 8, false),
      new THREE.MeshStandardMaterial({ color, roughness: 0.4, transparent: true, opacity: 0.85 }),
    );
    lane.trail.visible = ui.paths.checked;
    lane.rig.add(lane.trail);
  }

  function showQuery() {
    lanes.forEach((lane, i) => setTrajectory(lane, state.query[i], lane.color));
    state.restartAt = performance.now();
    refresh();
  }

  // --- UI ---------------------------------------------------------------------
  function refresh() {
    const ranking = state.mode === "rank";
    ui.round.textContent = state.mode === "best" ? "Robot's best guess" : `Round ${state.round + 1}`;
    ui.caption.innerHTML = "All 1,000 handovers in the learned feature space (2-D PCA view). " + CAPTIONS[state.method] +
      " ★ = the robot's current best guess.";
    ui.algos.forEach((b) => {
      const on = b.dataset.algo === state.method;
      b.classList.toggle("active", on);
      b.setAttribute("aria-checked", on);
      b.disabled = state.busy;
    });
    lanes.forEach((lane) => {
      const place = state.order.indexOf(lane.i);
      const rankText = place >= 0 ? PLACES[place] : "";
      const card = ui.cards[lane.i];
      card.classList.toggle("ranked", place >= 0);
      card.querySelector("[data-place]").textContent = rankText || "—";
      card.disabled = !ranking;
      lane.el.innerHTML = `<b>${lane.name}</b>${rankText ? `<span>${rankText}</span>` : ""}`;
      const visible = ranking || lane.i === 1;
      lane.group.visible = visible;
    });
    if (!ranking) lanes[1].el.innerHTML = `<b>★</b><span>best guess</span>`;
    lanes[1].plate.material.color.set(ranking ? lanes[1].color : BEST_COLOR);
    ui.submit.disabled = !ranking || state.order.length < K || state.busy;
    ui.clear.disabled = !ranking || !state.order.length;
    ui.best.disabled = state.round === 0 || state.busy;
    ui.best.textContent = ranking ? "Show robot's best guess" : "Back to ranking";
    ui.hint.textContent = !ranking
      ? "This is the handover the robot currently thinks you like most."
      : state.order.length === 0
        ? (state.round === 0 ? "Watch all three, then click your favourite first." : "New candidates! Click your favourite first.")
        : state.order.length < K ? "Now your second favourite." : "Happy with that? Submit your ranking.";
    drawSpace();
  }

  function pick(laneIdx) {
    if (state.mode !== "rank" || state.busy) return;
    if (state.order.includes(laneIdx)) {
      state.order = state.order.slice(0, state.order.indexOf(laneIdx)); // un-rank it and anything after
    } else {
      state.order.push(laneIdx);
      if (state.order.length === K - 1) state.order.push([0, 1, 2].find((p) => !state.order.includes(p)));
    }
    refresh();
  }

  ui.cards.forEach((card, i) => card.addEventListener("click", () => pick(i)));
  ui.clear.addEventListener("click", () => { state.order = []; refresh(); });
  ui.submit.addEventListener("click", () => {
    if (state.order.length < K || state.busy) return;
    state.busy = true;
    ui.thinking.hidden = false;
    refresh();
    setTimeout(() => { // let the "thinking" indicator paint first
      const ranked = state.order.map((p) => state.query[p]);
      state.belief.update(ranked.map((i) => features[i]));
      state.algorithm.tell(state.query, state.order);
      state.history.push(ranked);
      state.round += 1;
      state.order = [];
      state.query = state.algorithm.query(state.belief);
      state.busy = false;
      ui.thinking.hidden = true;
      showQuery();
      if (state.round === 10) toast("10 rankings — as many as each participant gave in our study. Peek at the robot's best guess!");
    }, 30);
  });
  function setMode(mode) {
    state.mode = mode;
    if (mode === "best") setTrajectory(lanes[1], bestIndex(), BEST_COLOR);
    else setTrajectory(lanes[1], state.query[1], lanes[1].color);
    state.restartAt = performance.now();
    refresh();
  }
  ui.best.addEventListener("click", () => {
    if (state.mode === "rank") { setMode("best"); setView("B"); }
    else { setMode("rank"); setView("All"); }
  });
  ui.algos.forEach((b) => b.addEventListener("click", () => {
    if (state.busy || b.dataset.algo === state.method) return;
    state.method = b.dataset.algo;
    reset();
    setView("All");
  }));
  ui.restart.addEventListener("click", () => { reset(); setView("All"); });
  ui.replay.addEventListener("click", () => { state.restartAt = performance.now(); });
  ui.speed.addEventListener("input", () => {
    const now = performance.now();
    const old = (REAL_DURATION / state.speed + HOLD) * 1000;
    const frac = ((now - state.restartAt) % old) / old;
    state.speed = parseFloat(ui.speed.value);
    state.restartAt = now - frac * (REAL_DURATION / state.speed + HOLD) * 1000;
    $("[data-speed-value]", container).textContent = `${state.speed}×`;
  });
  ui.paths.addEventListener("change", () => lanes.forEach((l) => { if (l.trail) l.trail.visible = ui.paths.checked; }));
  ui.views.forEach((b) => b.addEventListener("click", () => {
    if (state.mode === "best") setMode("rank"); // camera buttons always return to the three candidates
    setView(b.dataset.view);
  }));

  function toast(msg) {
    const t = $(".demo-toast", container);
    t.textContent = msg;
    t.hidden = false;
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => { t.hidden = true; }, 7000);
  }

  // click a robot in the viewport to rank it
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  let downAt = null;
  renderer.domElement.addEventListener("pointerdown", (e) => { downAt = [e.clientX, e.clientY]; });
  renderer.domElement.addEventListener("pointerup", (e) => {
    if (!downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 5) return; // was a drag
    const r = renderer.domElement.getBoundingClientRect();
    pointer.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    raycaster.setFromCamera(pointer, camera);
    const hit = raycaster.intersectObjects(lanes.map((l) => l.robot), true).find((h) => h.object.userData.lane !== undefined);
    if (hit) pick(hit.object.userData.lane);
  });

  // --- camera -----------------------------------------------------------------
  const camGoal = { pos: new THREE.Vector3(), target: new THREE.Vector3(), t: 1 };
  const camFrom = { pos: new THREE.Vector3(), target: new THREE.Vector3() };
  let currentView = "All";
  function setView(which, instant = false) {
    currentView = which;
    if (which === "All") {
      // fit all three lanes horizontally and the table-to-label height vertically
      const halfV = THREE.MathUtils.degToRad(camera.fov / 2);
      const halfH = Math.atan(Math.tan(halfV) * camera.aspect);
      const dist = 1.08 * Math.max((LANE_SPACING + 0.75) / Math.tan(halfH), 0.95 / Math.tan(halfV));
      const dir = new THREE.Vector3(1, -0.38, 0.5).normalize();
      camGoal.target.set(0.5, -0.3, 0.12);
      camGoal.pos.copy(camGoal.target).addScaledVector(dir, dist);
    } else {
      const y = (["A", "B", "C"].indexOf(which) - 1) * LANE_SPACING;
      const far = state.mode === "best" ? 1.3 : 1;
      // side-on to the robot-to-person line, so both the cup and the person are visible
      camGoal.target.set(0.65, y - 0.3, 0.2);
      camGoal.pos.set(0.65 + 0.85 * far, y - 0.3 + 1.15 * far, 0.2 + 0.8 * far);
    }
    camFrom.pos.copy(camera.position);
    camFrom.target.copy(controls.target);
    camGoal.t = instant ? 1 : 0;
    if (instant) { camera.position.copy(camGoal.pos); controls.target.copy(camGoal.target); }
    ui.views.forEach((b) => b.classList.toggle("active", b.dataset.view === which));
  }

  function resize() {
    const w = viewport.clientWidth, h = viewport.clientHeight;
    renderer.setSize(w, h);
    labelRenderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    if (currentView === "All") setView("All", true);
  }
  new ResizeObserver(resize).observe(viewport);
  resize();

  // --- plots (inline SVG) -----------------------------------------------------
  const SVG_NS = "http://www.w3.org/2000/svg";
  const xs = features2d.map((p) => p[0]), ys = features2d.map((p) => p[1]);
  const ext = { x0: Math.min(...xs), x1: Math.max(...xs), y0: Math.min(...ys), y1: Math.max(...ys) };
  const pad = 0.12 * Math.max(ext.x1 - ext.x0, ext.y1 - ext.y0);
  const W = 300, H = 230;
  const sx = (x) => ((x - ext.x0 + pad) / (ext.x1 - ext.x0 + 2 * pad)) * W;
  const sy = (y) => H - ((y - ext.y0 + pad) / (ext.y1 - ext.y0 + 2 * pad)) * H;
  const el = (tag, attrs) => { const e = document.createElementNS(SVG_NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); return e; };
  ui.space.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const dataLayer = el("g", {});
  features2d.forEach((p) => dataLayer.appendChild(el("circle", { cx: sx(p[0]), cy: sy(p[1]), r: 1.6, class: "pt-data" })));
  ui.space.appendChild(dataLayer);
  const dynLayer = el("g", {});
  ui.space.appendChild(dynLayer);

  function drawSpace() {
    dynLayer.replaceChildren();
    const dist = state.algorithm.searchDistribution();
    if (dist) {
      const m2 = pca.project(dist.mean);
      const PC = pca.P.map((p) => dist.cov.map((row) => row.reduce((s, v, j) => s + v * p[j], 0)));
      const c2 = pca.P.map((p) => PC.map((r) => r.reduce((s, v, j) => s + v * p[j], 0)));
      const { values, vectors } = eighSym(c2);
      const pts = [];
      for (let a = 0; a <= 64; a++) {
        const th = (a / 64) * 2 * Math.PI;
        const u = [2 * Math.sqrt(Math.max(values[0], 1e-12)) * Math.cos(th), 2 * Math.sqrt(Math.max(values[1], 1e-12)) * Math.sin(th)];
        pts.push(`${sx(m2[0] + vectors[0][0] * u[0] + vectors[0][1] * u[1])},${sy(m2[1] + vectors[1][0] * u[0] + vectors[1][1] * u[1])}`);
      }
      dynLayer.appendChild(el("polygon", { points: pts.join(" "), class: "search-dist" }));
    }
    if (state.algorithm.lastSamples.length > K) {
      for (const s of state.algorithm.lastSamples) {
        const p = pca.project(s);
        dynLayer.appendChild(el("circle", { cx: sx(p[0]), cy: sy(p[1]), r: 1.8, class: "pt-sample" }));
      }
    }
    state.history.forEach((h) => h.forEach((i) => {
      const p = features2d[i];
      dynLayer.appendChild(el("circle", { cx: sx(p[0]), cy: sy(p[1]), r: 2.4, class: "pt-past" }));
    }));
    if (state.mode === "rank") {
      lanes.forEach((lane) => {
        const p = features2d[state.query[lane.i]];
        dynLayer.appendChild(el("circle", { cx: sx(p[0]), cy: sy(p[1]), r: 6, fill: lane.color, class: "pt-query" }));
        const t = el("text", { x: sx(p[0]), y: sy(p[1]) + 3.2, class: "pt-label" });
        t.textContent = lane.name;
        dynLayer.appendChild(t);
      });
    }
    if (state.round > 0) {
      const p = features2d[bestIndex()];
      const star = (cx, cy, r) => Array.from({ length: 10 }, (_, k) => {
        const rr = k % 2 ? r * 0.45 : r, a = (k / 10) * 2 * Math.PI - Math.PI / 2;
        return `${cx + rr * Math.cos(a)},${cy + rr * Math.sin(a)}`;
      }).join(" ");
      dynLayer.appendChild(el("polygon", { points: star(sx(p[0]), sy(p[1]), 8), class: "pt-best" }));
    }
  }

  // --- animation loop -----------------------------------------------------------
  function frame(now) {
    const duration = (REAL_DURATION / state.speed) * 1000;
    const phase = Math.min(((now - state.restartAt) % (duration + HOLD * 1000)) / duration, 1);
    for (const lane of lanes) {
      if (!lane.group.visible || lane.index < 0) continue;
      setArm(lane.robot, sampleTraj(trajectories[lane.index], retime(phase), q));
    }
    if (camGoal.t < 1) {
      camGoal.t = Math.min(1, camGoal.t + 0.035);
      const e = 1 - (1 - camGoal.t) ** 3;
      camera.position.lerpVectors(camFrom.pos, camGoal.pos, e);
      controls.target.lerpVectors(camFrom.target, camGoal.target, e);
    }
    controls.update();
    renderer.render(scene, camera);
    labelRenderer.render(scene, camera);
    requestAnimationFrame(frame);
  }

  reset();
  setView("All", true);
  status.hidden = true;
  container.classList.add("ready");
  requestAnimationFrame(frame);
}

const container = document.querySelector("#demo");
if (container) {
  initDemo(container).catch((err) => {
    console.error(err);
    const s = container.querySelector(".demo-loading");
    s.hidden = false;
    s.textContent = `Sorry — the demo failed to load (${err.message}).`;
  });
}
