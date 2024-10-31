class KNearestNeighbor {
  constructor(t = []) {
    this.individuals = t;
  }
  inicializar(t, i, s) {
    (this.k = t), (this.data = i), (this.labels = s);
  }
  distance(t, i) {
    return Math.sqrt(t.map((t, s) => i[s] - t).reduce((t, i) => t + i * i, 0));
  }
  euclidean(t) {
    var i = [],
      s = this.individuals[0].length - 1;
    for (const e of this.individuals) {
      var a = e.slice(0, s);
      i.push(euclidean(t, a));
    }
    let e = Math.min(...i),
      n = [];
    for (const t in i) i[t] == e && n.push(this.individuals[t][s]);
    return [...new Set(n)];
  }
  manhattan(t) {
    var i = [],
      s = this.individuals[0].length - 1;
    for (const e of this.individuals) {
      var a = e.slice(0, s);
      i.push(manhattan(t, a));
    }
    let e = Math.min(...i),
      n = [];
    for (const t in i) i[t] == e && n.push(this.individuals[t][s]);
    return [...new Set(n)];
  }
  mapearGenerarDistancia(t) {
    const i = [];
    let s;
    for (let a = 0, e = this.data.length; a < e; a++) {
      const e = this.data[a],
        n = this.labels[a],
        l = this.distance(t, e);
      (!s || l < s) &&
        (i.push({ index: a, distance: l, label: n }),
        i.sort((t, i) => (t.distance < i.distance ? -1 : 1)),
        i.length > this.k && i.pop(),
        (s = i[i.length - 1].distance));
    }
    return i;
  }
  predecir(t) {
    const i = this.mapearGenerarDistancia(t).slice(0, this.k),
      s = i.reduce(
        (t, i) => Object.assign({}, t, { [i.label]: (t[i.label] || 0) + 1 }),
        {}
      );
    return {
      label: Object.keys(s)
        .map((t) => ({ label: t, count: s[t] }))
        .sort((t, i) => (t.count > i.count ? -1 : 1))[0].label,
      votosCounts: s,
      votos: i,
    };
  }
}
let m, b;
function MinimosCuadrados(t) {
  let e = SumaX(t),
    n = SumaY(t),
    u = SumaXY(t),
    l = SumaX2(t),
    r = t.length,
    o = u - (e * n) / r,
    f = Math.pow(e, 2) / r;
  (b = n / r - (m = o / (l - f)) * (e / r)), console.log(e, n, u, l, r);
}
function Predecir(t) {
  return [m * t + b, m, b];
}
function SumaX(t) {
  let e = t.length,
    n = 0;
  for (let u = 0; u < e; u++) n += t[u][0];
  return n;
}
function SumaY(t) {
  let e = t.length,
    n = 0;
  for (let u = 0; u < e; u++) n += t[u][1];
  return n;
}
function SumaXY(t) {
  let e = t.length,
    n = 0,
    u = [];
  for (let n = 0; n < e; n++) {
    let e = t[n][0],
      l = t[n][1];
    u.push(e * l);
  }
  for (let t = 0; t < e; t++) n += u[t];
  return n;
}
function SumaX2(t) {
  let e = t.length,
    n = 0,
    u = [];
  for (let n = 0; n < e; n++) {
    let e = t[n][0];
    u.push(e * e);
  }
  for (let t = 0; t < e; t++) n += u[t];
  return n;
}
var tabla = new Array(
    new Array("Outlook", "Temperature", "Humidity", "Windy", "Class"),
    new Array("sunny", "hot", "high", "false", "N"),
    new Array("sunny", "hot", "high", "true", "N"),
    new Array("overcast", "hot", "high", "false", "P"),
    new Array("rain", "mild", "high", "false", "P"),
    new Array("rain", "cool", "normal", "false", "P"),
    new Array("rain", "cool", "normal", "true", "N"),
    new Array("overcast", "cool", "normal", "true", "P"),
    new Array("sunny", "mild", "high", "false", "N"),
    new Array("sunny", "cool", "normal", "false", "P"),
    new Array("rain", "mild", "normal", "false", "P"),
    new Array("sunny", "mild", "normal", "true", "P"),
    new Array("overcast", "mild", "high", "true", "P"),
    new Array("overcast", "hot", "normal", "false", "P"),
    new Array("rain", "mild", "high", "true", "N")
  ),
  ejemplo2 = new Array(
    new Array("Empleo", "Ingresos", "Credito"),
    new Array("si", "3000", "P"),
    new Array("si", "1000", "N"),
    new Array("no", "0", "N"),
    new Array("si", "1500", "P"),
    new Array("si", "2000", "P"),
    new Array("no", "0", "N")
  ),
  ejemplo3 = new Array(
    new Array(
      "Presion Arterial",
      "Urea en Sangre",
      "Gota",
      "Hipotiroidismo",
      "Admin"
    ),
    new Array("Alta", "Alta", "Si", "No", "No"),
    new Array("Alta", "Alta", "Si", "Si", "No"),
    new Array("Normal", "Alta", "Si", "No", "Si"),
    new Array("Baja", "Normal", "Si", "No", "Si"),
    new Array("Baja", "Baja", "No", "No", "Si"),
    new Array("Baja", "Baja", "No", "Si", "No"),
    new Array("Normal", "Baja", "No", "Si", "Si"),
    new Array("Alta", "Normal", "Si", "No", "No"),
    new Array("Alta", "Baja", "No", "No", "Si"),
    new Array("Baja", "Normal", "No", "No", "Si"),
    new Array("Alta", "Normal", "No", "Si", "Si"),
    new Array("Normal", "Normal", "Si", "Si", "Si"),
    new Array("Normal", "Alta", "No", "No", "Si"),
    new Array("Baja", "Normal", "Si", "Si", "No")
  ),
  ejemplo1 = new Array(
    new Array("Outlook", "Temperature", "Humidity", "Windy", "Class"),
    new Array("sunny", "hot", "high", "false", "N"),
    new Array("sunny", "hot", "high", "true", "N"),
    new Array("overcast", "hot", "high", "false", "P"),
    new Array("rain", "mild", "high", "false", "P"),
    new Array("rain", "cool", "normal", "false", "P"),
    new Array("rain", "cool", "normal", "true", "N"),
    new Array("overcast", "cool", "normal", "true", "P"),
    new Array("sunny", "mild", "high", "false", "N"),
    new Array("sunny", "cool", "normal", "false", "P"),
    new Array("rain", "mild", "normal", "false", "P"),
    new Array("sunny", "mild", "normal", "true", "P"),
    new Array("overcast", "mild", "high", "true", "P"),
    new Array("overcast", "hot", "normal", "false", "P"),
    new Array("rain", "mild", "high", "true", "N")
  ),
  varGanar = "P",
  varPerder = "N",
  totalGanar = 0,
  totalPerder = 0,
  total = 0,
  varIG = 0,
  varEntropia = 0,
  varGanancia = 0;
function cantidad(e, r) {
  let a = 0;
  for (let n = 1; n < r.length; n++) e == r[n][r[0].length - 1] && a++;
  return a;
}
function cantidadPosicion(e, r, a, n) {
  let t = 0;
  for (let o = 1; o < n.length; o++)
    e == n[o][a] && r == n[o][n[0].length - 1] && t++;
  return t;
}
function atributo(e) {
  (this.nombre = e), (this.ganar = 0), (this.perder = 0), (this.info = 0);
}
function listaAtributo(e, r) {
  let a = new Set();
  for (let n = 1; n < r.length; n++) a.add(r[n][e]);
  let n = new Array(),
    t = 0;
  for (let e of a.values()) {
    let r = new atributo(e);
    n[t++] = r;
  }
  return n;
}
function revisionNulo(e) {
  return isNaN(e) ? 0 : e;
}
function informacionGeneral() {
  let e = 0,
    r = totalGanar / (totalGanar + totalPerder),
    a = totalPerder / (totalGanar + totalPerder);
  return (e = -r * Math.log2(r) - a * Math.log2(a));
}
function entropia(e, r) {
  let a = listaAtributo(e, r);
  for (let n = 0; n < a.length; n++)
    for (let t = 1; t < r.length; t++) {
      let o = r[t][e];
      o == a[n].nombre && varGanar == r[t][r[0].length - 1]
        ? a[n].ganar++
        : o == a[n].nombre &&
          varPerder == r[t][r[0].length - 1] &&
          a[n].perder++;
    }
  for (let e = 0; e < a.length; e++) {
    let r = a[e].ganar / (a[e].ganar + a[e].perder),
      n = a[e].perder / (a[e].ganar + a[e].perder),
      t = -r * Math.log2(r) - n * Math.log2(n);
    a[e].info = revisionNulo(t);
  }
  let n = 0;
  for (let e = 0; e < a.length; e++) {
    n += ((a[e].ganar + a[e].perder) / (totalGanar + totalPerder)) * a[e].info;
  }
  return n;
}
function ganancia(e, r) {
  return e - r;
}
function id3(e) {
  (totalGanar = cantidad(varGanar, e)),
    (totalPerder = cantidad(varPerder, e)),
    (total = totalGanar + totalPerder),
    (varIG = informacionGeneral());
  let r = 0,
    a = "";
  for (let n = 1; n < e[0].length; n++)
    (varEntropia = entropia(n - 1, e)),
      (varGanancia = ganancia(varIG, varEntropia)) > r &&
        ((r = varGanancia), (a = e[0][n - 1]));
  return a;
}
function convertirTabla(e, r) {
  let a = new Array(),
    n = 0,
    t = new Array();
  for (let e = 1; e < r[0].length; e++) t[e - 1] = r[0][e];
  a[n++] = t;
  for (let t = 0; t < r.length; t++)
    if (e == r[t][0]) {
      let e = new Array();
      for (let a = 1; a < r[0].length; a++) e[a - 1] = r[t][a];
      a[n++] = e;
    }
  return a;
}
function obtenerPosicionEncabezado(e, r) {
  for (let a = 0; a < r[0].length; a++) if (e == r[0][a]) return a;
  return 0;
}
function revisarAmbiguedad(e, r, a) {
  let n = 0,
    t = 0;
  for (let o = 0; o < e.length; o++)
    e[o][a] == r && e[o][e[0].length - 1] == varGanar
      ? n++
      : e[o][a] == r && e[o][e[0].length - 1] == varPerder && t++;
  return (0 != n && 0 != t) || (0 == n ? varPerder : varGanar);
}
let iterador = 1;
function getNodo(e, r, a, n) {
  let t = "";
  for (let o = 0; o < r.length; o++) {
    let l = iterador++;
    t += e + "--" + l + ' [label="' + r[o].nombre + '"];';
    let i = obtenerPosicionEncabezado(n, a),
      d = revisarAmbiguedad(a, r[o].nombre, i),
      m = varGanar;
    if (1 == d) {
      let e = convertirTabla(r[o].nombre, a);
      t += getNodo(
        l,
        listaAtributo(obtenerPosicionEncabezado((m = id3(e)), e), e),
        e,
        m
      );
    } else d == varPerder && (m = varPerder);
    t += l + ' [label="' + m + '"];';
  }
  return t;
}
function treeDirec() {
  var e = "graph {";
  let r = id3(tabla);
  e += iterador++ + ' [label="' + r + '"];';
  let a = listaAtributo(obtenerPosicionEncabezado(r, tabla), tabla);
  return (
    (e += getNodo(iterador - 1, a, tabla, r)), (e += "}"), console.log(e), e
  );
}
function Mejemplo1() {
  var e = document.getElementById("b"),
    r = document.getElementById("mynetwork");
  e.removeChild(r);
  var a = document.createElement("div");
  (a.id = "mynetwork"),
    document.getElementById("b").appendChild(a),
    (tabla = ejemplo3);
  var n = document.getElementById("mynetwork"),
    t = treeDirec(),
    o = vis.network.convertDot(t),
    l = { nodes: o.nodes, edges: o.edges };
  new vis.Network(n, l, {
    layout: {
      hierarchical: {
        levelSeparation: 100,
        nodeSpacing: 100,
        parentCentralization: !0,
        direction: "UD",
        sortMethod: "directed",
      },
    },
  });
}
function Mejemplo2() {
  var e = document.getElementById("b"),
    r = document.getElementById("mynetwork");
  e.removeChild(r);
  var a = document.createElement("div");
  (a.id = "mynetwork"),
    document.getElementById("b").appendChild(a),
    (tabla = ejemplo2);
  var n = document.getElementById("mynetwork"),
    t = treeDirec(),
    o = vis.network.convertDot(t),
    l = { nodes: o.nodes, edges: o.edges };
  new vis.Network(n, l, {
    layout: {
      hierarchical: {
        levelSeparation: 100,
        nodeSpacing: 100,
        parentCentralization: !0,
        direction: "UD",
        sortMethod: "directed",
      },
    },
  });
}
function Mejemplo3() {
  var e = document.getElementById("b"),
    r = document.getElementById("mynetwork");
  e.removeChild(r);
  var a = document.createElement("div");
  (a.id = "mynetwork"),
    document.getElementById("b").appendChild(a),
    (tabla = ejemplo1);
  var n = document.getElementById("mynetwork"),
    t = treeDirec(),
    o = vis.network.convertDot(t),
    l = { nodes: o.nodes, edges: o.edges };
  new vis.Network(n, l, {
    layout: {
      hierarchical: {
        levelSeparation: 100,
        nodeSpacing: 100,
        parentCentralization: !0,
        direction: "UD",
        sortMethod: "directed",
      },
    },
  });
}
class Kmeans_G13 {
  constructor(t) {
    (this.canvas = t.canvas),
      (this.context = this.canvas.getContext("2d")),
      (this.width = this.canvas.width),
      (this.height = this.canvas.height),
      (this.k = t.k),
      (this.data = t.data),
      (this.assignments = []),
      (this.extents = this.Dimensions()),
      (this.ranges = this.dataExtentRanges()),
      (this.means = this.seeds()),
      (this.clusterColors = this.clusterColors()),
      (this.iterations = t.iterations),
      (this.context.fillStyle = "#FFFFFF"),
      this.context.fillRect(0, 0, this.width, this.height),
      this.draw(),
      (this.drawDelay = 50),
      this.run();
  }
  Dimensions() {
    for (var t = [], s = 0; s < this.data.length; s++)
      for (var i = this.data[s], e = 0; e < i.length; e++)
        t[e] || (t[e] = { min: 1e3, max: 0 }),
          i[e] < t[e].min && (t[e].min = i[e]),
          i[e] > t[e].max && (t[e].max = i[e]);
    return t;
  }
  dataExtentRanges() {
    for (var t = [], s = 0; s < this.extents.length; s++)
      t[s] = this.extents[s].max - this.extents[s].min;
    return t;
  }
  seeds() {
    for (var t = []; this.k--; ) {
      for (var s = [], i = 0; i < this.extents.length; i++)
        s[i] = this.extents[i].min + Math.random() * this.ranges[i];
      t.push(s);
    }
    return t;
  }
  assignClusterToDataPoints() {
    for (var t = [], s = 0; s < this.data.length; s++) {
      for (var i = this.data[s], e = [], h = 0; h < this.means.length; h++) {
        for (var n = this.means[h], a = 0, r = 0; r < i.length; r++) {
          var o = i[r] - n[r];
          a += o = Math.pow(o, 2);
        }
        e[h] = Math.sqrt(a);
      }
      t[s] = e.indexOf(Math.min.apply(null, e));
    }
    return t;
  }
  moveMeans() {
    var t,
      s,
      i,
      e,
      h = this.fillArray(this.means.length, 0),
      n = this.fillArray(this.means.length, 0),
      a = !1;
    for (t = 0; t < this.means.length; t++)
      h[t] = this.fillArray(this.means[t].length, 0);
    for (var r = 0; r < this.assignments.length; r++) {
      s = this.assignments[r];
      var o = this.data[r],
        l = this.means[s];
      for (n[s]++, i = 0; i < l.length; i++) h[s][i] += o[i];
    }
    for (s = 0; s < h.length; s++)
      if (0 !== n[s])
        for (i = 0; i < h[s].length; i++)
          (h[s][i] /= n[s]), (h[s][i] = Math.round(100 * h[s][i]) / 100);
      else
        for (h[s] = this.means[s], i = 0; i < this.extents.length; i++)
          h[s][i] = this.extents[i].min + Math.random() * this.ranges[i];
    if (this.means.toString() !== h.toString())
      for (a = !0, s = 0; s < h.length; s++)
        for (i = 0; i < h[s].length; i++)
          if (((e = h[s][i] - this.means[s][i]), Math.abs(e) > 0.1)) {
            (this.means[s][i] += e / 10),
              (this.means[s][i] = Math.round(100 * this.means[s][i]) / 100);
          } else this.means[s][i] = h[s][i];
    return a;
  }
  run() {
    ++this.iterations,
      (this.assignments = this.assignClusterToDataPoints()),
      this.moveMeans() &&
        (this.draw(),
        (this.timer = setTimeout(this.run.bind(this), this.drawDelay)));
  }
  draw() {
    var t, s;
    for (
      this.context.fillStyle = "rgba(255,255,255, 0.2)",
        this.context.fillRect(0, 0, this.width, this.height),
        s = 0;
      s < this.assignments.length;
      s++
    ) {
      var i = this.assignments[s];
      t = this.data[s];
      var e = this.means[i];
      (this.context.globalAlpha = 0.1),
        this.context.save(),
        this.context.beginPath(),
        this.context.moveTo(
          (t[0] - this.extents[0].min + 1) *
            (this.width / (this.ranges[0] + 2)),
          (t[1] - this.extents[1].min + 1) *
            (this.height / (this.ranges[1] + 2))
        ),
        this.context.lineTo(
          (e[0] - this.extents[0].min + 1) *
            (this.width / (this.ranges[0] + 2)),
          (e[1] - this.extents[1].min + 1) *
            (this.height / (this.ranges[1] + 2))
        ),
        (this.context.strokeStyle = "black"),
        this.context.stroke(),
        this.context.restore();
    }
    for (s = 0; s < this.data.length; s++)
      this.context.save(),
        (t = this.data[s]),
        (this.context.globalAlpha = 1),
        this.context.translate(
          (t[0] - this.extents[0].min + 1) *
            (this.width / (this.ranges[0] + 2)),
          (t[1] - this.extents[1].min + 1) * (this.width / (this.ranges[1] + 2))
        ),
        this.context.beginPath(),
        this.context.arc(0, 0, 5, 0, 2 * Math.PI, !0),
        (this.context.strokeStyle = this.clusterColor(this.assignments[s])),
        this.context.stroke(),
        this.context.closePath(),
        this.context.restore();
    for (s = 0; s < this.means.length; s++)
      this.context.save(),
        (t = this.means[s]),
        (this.context.globalAlpha = 0.5),
        (this.context.fillStyle = this.clusterColor(s)),
        this.context.translate(
          (t[0] - this.extents[0].min + 1) *
            (this.width / (this.ranges[0] + 2)),
          (t[1] - this.extents[1].min + 1) * (this.width / (this.ranges[1] + 2))
        ),
        this.context.beginPath(),
        this.context.arc(0, 0, 5, 0, 2 * Math.PI, !0),
        this.context.fill(),
        this.context.closePath(),
        this.context.restore();
  }
  clusterColors() {
    for (var t = [], s = 0; s < this.data.length; s++)
      t.push("#" + ((Math.random() * (1 << 24)) | 0).toString(16));
    return t;
  }
  clusterColor(t) {
    return this.clusterColors[t];
  }
  fillArray(t, s) {
    return Array.apply(null, Array(t)).map(function () {
      return s;
    });
  }
}
const M = 9,
  x = [7, 1, 10, 5, 4, 3, 13, 10, 2],
  y = [2, 9, 2, 5, 7, 11, 2, 5, 14],
  tasaAprendisaje = 5e-4;
let theta1 = 0,
  thetaInicial = 0;
const h = (t) => thetaInicial + theta1 * t,
  learn = (t) => {
    let e = 0,
      a = 0;
    for (let t = 0; t < 9; t++)
      (e += h(x[t]) - y[t]), (a += (h(x[t]) - y[t]) * x[t]);
    (thetaInicial -= (t / 9) * e), (theta1 -= (t / 9) * a);
  },
  cost = () => {
    let t = 0;
    for (let e = 0; e < 9; e++) t += Math.pow(h(x[e]) - y[e], 2);
    return t / 18;
  };
let iteration = 0;
function GradienteDescendente() {
  var t,
    e = [];
  for (t = 0; t < 18e4; t++) {
    if ((learn(tasaAprendisaje), t % 6e3 == 0)) {
      let a = {
        Iteracion: t,
        Funcion:
          "f(x) = " + thetaInicial.toFixed(2) + " + " + theta1.toFixed(2) + "x",
      };
      e.push(a);
    }
    t++;
  }
  let a = {
    Iteracion: t,
    Funcion:
      "f(x) = " + thetaInicial.toFixed(2) + " + " + theta1.toFixed(2) + "x",
  };
  return e.push(a), e;
}
function LineLeastSquares(e, r) {
  var n = 0,
    t = 0,
    a = 0,
    o = 0,
    s = 0,
    l = 0,
    h = 0,
    g = [];
  if (e.length != r.length)
    throw new Error("Los array ingresados deben ser del mismo tamaño");
  for (let g = 0; g < e.length; g++)
    (o += l = e[g]), (s += h = r[g]), (n += l * l), (t += l * h), a++;
  var i = (a * t - o * s) / (a * n - o * o),
    f = s / a - (i * o) / a;
  for (let r = 0; r < e.length; r++) (h = (l = e[r]) * i + f), g.push(h);
  return [i, f, g];
}
class G8_Kmeans {
  constructor(t) {
    (this.canvas = t.canvas),
      (this.context = this.canvas.getContext("2d")),
      (this.width = this.canvas.width),
      (this.height = this.canvas.height),
      (this.k = t.k),
      (this.data = t.data),
      (this.assignments = []),
      (this.extents = this.dataDimensionExtents()),
      (this.ranges = this.dataExtentRanges()),
      (this.means = this.seeds()),
      (this.clusterColors = this.clusterColors()),
      (this.iterations = 0),
      (this.context.fillStyle = "rgb(255,255,255)"),
      this.context.fillRect(0, 0, this.width, this.height),
      this.draw(),
      (this.drawDelay = 20),
      this.run();
  }
  dataDimensionExtents() {
    for (var t = [], s = 0; s < this.data.length; s++)
      for (var i = this.data[s], e = 0; e < i.length; e++)
        t[e] || (t[e] = { min: 1e3, max: 0 }),
          i[e] < t[e].min && (t[e].min = i[e]),
          i[e] > t[e].max && (t[e].max = i[e]);
    return t;
  }
  dataExtentRanges() {
    for (var t = [], s = 0; s < this.extents.length; s++)
      t[s] = this.extents[s].max - this.extents[s].min;
    return t;
  }
  seeds() {
    for (var t = []; this.k--; ) {
      for (var s = [], i = 0; i < this.extents.length; i++)
        s[i] = this.extents[i].min + Math.random() * this.ranges[i];
      t.push(s);
    }
    return t;
  }
  assignClusterToDataPoints() {
    for (var t = [], s = 0; s < this.data.length; s++) {
      for (var i = this.data[s], e = [], h = 0; h < this.means.length; h++) {
        for (var n = this.means[h], a = 0, r = 0; r < i.length; r++) {
          var o = i[r] - n[r];
          a += o = Math.pow(o, 2);
        }
        e[h] = Math.sqrt(a);
      }
      t[s] = e.indexOf(Math.min.apply(null, e));
    }
    return t;
  }
  moveMeans() {
    var t,
      s,
      i,
      e,
      h = this.fillArray(this.means.length, 0),
      n = this.fillArray(this.means.length, 0),
      a = !1;
    for (t = 0; t < this.means.length; t++)
      h[t] = this.fillArray(this.means[t].length, 0);
    for (var r = 0; r < this.assignments.length; r++) {
      s = this.assignments[r];
      var o = this.data[r],
        l = this.means[s];
      for (n[s]++, i = 0; i < l.length; i++) h[s][i] += o[i];
    }
    for (s = 0; s < h.length; s++)
      if (0 !== n[s])
        for (i = 0; i < h[s].length; i++)
          (h[s][i] /= n[s]), (h[s][i] = Math.round(100 * h[s][i]) / 100);
      else
        for (h[s] = this.means[s], i = 0; i < this.extents.length; i++)
          h[s][i] = this.extents[i].min + Math.random() * this.ranges[i];
    if (this.means.toString() !== h.toString())
      for (a = !0, s = 0; s < h.length; s++)
        for (i = 0; i < h[s].length; i++)
          if (((e = h[s][i] - this.means[s][i]), Math.abs(e) > 0.1)) {
            (this.means[s][i] += e / 10),
              (this.means[s][i] = Math.round(100 * this.means[s][i]) / 100);
          } else this.means[s][i] = h[s][i];
    return a;
  }
  run() {
    ++this.iterations,
      (this.assignments = this.assignClusterToDataPoints()),
      this.moveMeans()
        ? (this.draw(),
          (this.timer = setTimeout(this.run.bind(this), this.drawDelay)))
        : console.log("Iteration took for completion: " + this.iterations);
  }
  draw() {
    var t, s;
    for (
      this.context.fillStyle = "rgba(255,255,255, 0.2)",
        this.context.fillRect(0, 0, this.width, this.height),
        s = 0;
      s < this.assignments.length;
      s++
    ) {
      var i = this.assignments[s];
      t = this.data[s];
      var e = this.means[i];
      (this.context.globalAlpha = 0.1),
        this.context.save(),
        this.context.beginPath(),
        this.context.moveTo(
          (t[0] - this.extents[0].min + 1) *
            (this.width / (this.ranges[0] + 2)),
          (t[1] - this.extents[1].min + 1) *
            (this.height / (this.ranges[1] + 2))
        ),
        this.context.lineTo(
          (e[0] - this.extents[0].min + 1) *
            (this.width / (this.ranges[0] + 2)),
          (e[1] - this.extents[1].min + 1) *
            (this.height / (this.ranges[1] + 2))
        ),
        (this.context.strokeStyle = "black"),
        this.context.stroke(),
        this.context.restore();
    }
    for (s = 0; s < this.data.length; s++)
      this.context.save(),
        (t = this.data[s]),
        (this.context.globalAlpha = 1),
        this.context.translate(
          (t[0] - this.extents[0].min + 1) *
            (this.width / (this.ranges[0] + 2)),
          (t[1] - this.extents[1].min + 1) * (this.width / (this.ranges[1] + 2))
        ),
        this.context.beginPath(),
        this.context.arc(0, 0, 5, 0, 2 * Math.PI, !0),
        (this.context.strokeStyle = this.clusterColor(this.assignments[s])),
        this.context.stroke(),
        this.context.closePath(),
        this.context.restore();
    for (s = 0; s < this.means.length; s++)
      this.context.save(),
        (t = this.means[s]),
        (this.context.globalAlpha = 0.5),
        (this.context.fillStyle = this.clusterColor(s)),
        this.context.translate(
          (t[0] - this.extents[0].min + 1) *
            (this.width / (this.ranges[0] + 2)),
          (t[1] - this.extents[1].min + 1) * (this.width / (this.ranges[1] + 2))
        ),
        this.context.beginPath(),
        this.context.arc(0, 0, 5, 0, 2 * Math.PI, !0),
        this.context.fill(),
        this.context.closePath(),
        this.context.restore();
  }
  clusterColors() {
    for (var t = [], s = 0; s < this.data.length; s++)
      t.push("#" + ((Math.random() * (1 << 24)) | 0).toString(16));
    return t;
  }
  clusterColor(t) {
    return this.clusterColors[t];
  }
  fillArray(t, s) {
    return Array.apply(null, Array(t)).map(function () {
      return s;
    });
  }
}
class BayesMethod {
  constructor() {
    (this.attributes = []),
      (this.classes = []),
      (this.frecuencyTables = []),
      (this.attributeNames = []),
      (this.className = null);
  }
  addAttribute(t, e) {
    if (e && this.attributeNames.includes(e)) return !1;
    if (t) {
      if (!Array.isArray(t)) return !1;
      {
        let e = !0;
        if ((t.forEach((t) => (e = e && t !== Object(t))), !e)) return !1;
      }
      this.attributes.push(t), e && this.attributeNames.push(e);
    }
    return !0;
  }
  addClass(t, e) {
    if (this.class) return !1;
    if (t) {
      if (!Array.isArray(t)) return !1;
      {
        let e = !0;
        if ((t.forEach((t) => (e = e && t !== Object(t))), !e)) return !1;
      }
      (this.classes = t), e && (this.className = e);
    }
    return !0;
  }
  train() {
    if (!this.isModelValid()) return !1;
    this.attributes.forEach((t, e) => {
      let r = this.toFrecuencyTable(t);
      r && this.frecuencyTables.push(r);
    });
    var t = this.toFrecuencyTable(this.classes);
    return (
      this.frecuencyTables.push(t),
      this.attributes.forEach((e, r) => {
        let s = this.frecuencyTables[r];
        var i = [];
        s.values.forEach((r, o) => {
          var n = [];
          t.values.forEach((t, i) => {
            n.push(0),
              e.forEach((e, s) => {
                e == r && this.classes[s] == t && (n[n.length - 1] += 1);
              }),
              (n[n.length - 1] = n[n.length - 1] / s.frecuencies[o]);
          }),
            i.push(n);
        }),
          (s.valueClassProbabilities = i);
      }),
      !0
    );
  }
  probability(t, e, r) {
    var s = this.attributeNames.findIndex((e) => e === t);
    if (-1 == s) return null;
    var i = this.frecuencyTables[s],
      o = i.values.findIndex((t) => t == e);
    if (-1 == o) return null;
    var n = this.frecuencyTables[this.frecuencyTables.length - 1],
      a = n.values.findIndex((t) => t == r);
    return (
      (i.valueClassProbabilities[o][a] * i.probabilities[o]) /
      n.probabilities[a]
    );
  }
  predict(t, e = null) {
    var r = [],
      s = this.frecuencyTables[this.frecuencyTables.length - 1];
    if (
      (s.values.forEach((e, i) => {
        var o = s.probabilities[i],
          n = 1;
        t.forEach((t, r) => {
          if (null != t) {
            var s = this.probability(this.attributeNames[r], t, e);
            null != s && (n *= s);
          }
        }),
          r.push(o * n);
      }),
      null != e)
    ) {
      var i = s.values.findIndex((t) => t === e);
      return -1 == i ? [e, null] : [e, r[i]];
    }
    if (r.length > 1) {
      var o = 0,
        n = r[0];
      return (
        r.forEach((t, e) => {
          t > n && ((o = e), (n = t));
        }),
        [this.classes[o], n]
      );
    }
    return null;
  }
  isModelValid() {
    if (!this.classes) return !1;
    if (!this.attributes) return !1;
    {
      let t = this.attributes[0].length,
        e = !0;
      if (
        (this.attributes.forEach((r) => (e = e && r.length == t)),
        !(e = e && this.classes.length))
      )
        return (
          console.log(
            "attributes and class must have the same ammount of elements"
          ),
          !1
        );
    }
    return !0;
  }
  toFrecuencyTable(t) {
    if (!Array.isArray(t)) return !1;
    {
      let e = !0;
      if ((t.forEach((t) => (e = e && t !== Object(t))), !e)) return !1;
    }
    var e = [],
      r = [],
      s = [];
    return (
      t.forEach((t, s) => {
        var i = e.findIndex((e) => e === t);
        i > -1 ? (r[i] += 1) : (e.push(t), r.push(1));
      }),
      r.forEach((e) => s.push(e / t.length)),
      { values: e, frecuencies: r, probabilities: s }
    );
  }
}
class NodeTree {
  constructor(t = null, e = "", r = []) {
    (this.id = Math.random().toString(15).substr(3, 12)),
      (this.tag = e),
      (this.value = t),
      (this.childs = r);
  }
}
class Feature {
  constructor(t, e, r) {
    (this.attribute = t),
      (this.entropy = -1),
      (this.gain = -1),
      (this.primaryCount = 0),
      (this.secondaryCount = 0),
      (this.primaryPosibility = e),
      (this.secondPosibility = r);
  }
  updateFeature(t) {
    if (t === this.primaryPosibility) this.primaryCount += 1;
    else {
      if (t !== this.secondPosibility) return !1;
      this.secondaryCount += 1;
    }
    return (
      (this.entropy = this.calculateEntropy(
        this.primaryCount,
        this.secondaryCount
      )),
      !0
    );
  }
  calculateEntropy(t, e) {
    let r = -1;
    return 0 == t || 0 == e
      ? 0
      : ((r = (-t / (t + e)) * Math.log2(t / (t + e))),
        (r += (-e / (t + e)) * Math.log2(e / (t + e))));
  }
}
class Attribute {
  constructor(t) {
    (this.attribute = t),
      (this.features = []),
      (this.infoEntropy = -1),
      (this.gain = -1),
      (this.index = -1);
  }
}
class DecisionTreeID3 {
  constructor(t = []) {
    (this.dataset = t),
      (this.generalEntropy = -1),
      (this.primaryCount = -1),
      (this.secondaryCount = -1),
      (this.primaryPosibility = ""),
      (this.secondPosibility = ""),
      (this.root = null);
  }
  calculateEntropy(t, e) {
    let r = -1;
    return 0 == t || 0 == e
      ? 0
      : ((r = (-t / (t + e)) * Math.log2(t / (t + e))),
        (r += (-e / (t + e)) * Math.log2(e / (t + e))));
  }
  train(t, e = 0) {
    let r = t[0].length - 1;
    this.calculateGeneralEntropy(t, r);
    let s = t[0].length,
      i = [];
    for (let o = e; o < s; o++) {
      if (o === r) continue;
      let e = new Attribute(t[0][o]);
      (e.index = o),
        (e.features = this.classifierFeatures(t, o, r)),
        (e.infoEntropy = this.calculateInformationEntropy(e.features)),
        (e.gain = this.calculateGain(this.generalEntropy, e.infoEntropy)),
        i.push(e);
    }
    if (0 == i.length) return null;
    let o = this.selectBestFeature(i),
      n = new NodeTree(i[o].attribute);
    return (
      i[o].features.map((r) => {
        let s = new NodeTree(null);
        if (0 == r.entropy)
          s.value =
            0 == r.primaryCount ? r.secondPosibility : r.primaryPosibility;
        else {
          let n = t.filter((t, e) => t[i[o].index] === r.attribute || 0 == e);
          e < 4 && n.length > 2 && (s = this.train(n, e + 1));
        }
        (s.tag = r.attribute), n.childs.push(s);
      }),
      n
    );
  }
  predict(t, e) {
    return this.recursivePredict(t, e);
  }
  recursivePredict(t, e) {
    if (0 == e.childs.length) return e;
    for (let r = 0; r < t[0].length; r++)
      if (t[0][r] === e.value)
        for (let s = 0; s < e.childs.length; s++)
          if (e.childs[s].tag === t[1][r])
            return this.recursivePredict(t, e.childs[s]);
    return null;
  }
  calculateGeneralEntropy(t, e) {
    let r = { tag: "", count: 0 },
      s = { tag: "", count: 0 },
      i = !1;
    return (
      t.map((t) => {
        i
          ? r.tag
            ? s.tag || t[e] == r.tag
              ? r.tag === t[e]
                ? (r.count += 1)
                : s.tag === t[e] && (s.count += 1)
              : ((s.tag = t[e]), (s.count += 1))
            : ((r.tag = t[e]), (r.count += 1))
          : (i = !0);
      }),
      (this.primaryPosibility = r.tag),
      (this.secondPosibility = s.tag),
      (this.primaryCount = r.count),
      (this.secondaryCount = s.count),
      (this.generalEntropy = this.calculateEntropy(r.count, s.count)),
      this.generalEntropy
    );
  }
  classifierFeatures(t, e, r) {
    let s = [],
      i = !1;
    return (
      t.map((t) => {
        if (i) {
          let i = s.findIndex((r) => r.attribute === t[e]);
          if (i > -1) s[i].updateFeature(t[r]);
          else {
            let i = new Feature(
              t[e],
              this.primaryPosibility,
              this.secondPosibility
            );
            i.updateFeature(t[r]), s.push(i);
          }
        } else i = !0;
      }),
      s
    );
  }
  calculateInformationEntropy(t) {
    let e = 0;
    return (
      t.map((t) => {
        e +=
          ((t.primaryCount + t.secondaryCount) /
            (this.primaryCount + this.secondaryCount)) *
          t.entropy;
      }),
      e
    );
  }
  calculateGain(t, e) {
    return t - e;
  }
  selectBestFeature(t) {
    let e = -1,
      r = -1e3;
    return (
      t.map((t, s) => {
        t.gain > r && ((r = t.gain), (e = s));
      }),
      e
    );
  }
  generateDotString(t) {
    let e = "{";
    return (e += this.recursiveDotString(t)) + "}";
  }
  recursiveDotString(t, e = "") {
    let r = "";
    return t
      ? ((r += `${t.id} [label="${t.value}"];`),
        (r += e ? `${e}--${t.id}` : ""),
        (r += t.tag ? `[label="${t.tag}"];` : ""),
        t.childs.map((e) => {
          r += this.recursiveDotString(e, t.id);
        }),
        r)
      : "";
  }
}
class FuzzySet {
  constructor(t = []) {
    (this.individuals = t), (this.minsAndMaxs = this.calculateMinsAndMax());
  }
  calculateMinsAndMax() {
    var t = [[], [], []],
      e = this.individuals[0].length - 1;
    for (let r = 0; r < this.individuals.length; r++)
      for (let s = 0; s < e; s++)
        null == t[s][0]
          ? (t[s][0] = this.individuals[r][s])
          : (t[s][0] = Math.min(t[s][0], this.individuals[r][s])),
          null == t[s][1]
            ? (t[s][1] = this.individuals[r][s])
            : (t[s][1] = Math.max(t[s][1], this.individuals[r][s]));
    return t;
  }
  normalization() {
    var t = [],
      e = this.individuals[0].length - 1;
    for (let r = 0; r < this.individuals.length; r++) {
      let s = [];
      for (let t = 0; t < e; t++)
        this.individuals[r][t] <= this.minsAndMaxs[t][0]
          ? s.push(0)
          : this.individuals[r][t] >= this.minsAndMaxs[t][1]
          ? s.push(1)
          : s.push(
              (
                (this.individuals[r][t] - this.minsAndMaxs[t][0]) /
                (this.minsAndMaxs[t][1] - this.minsAndMaxs[t][0])
              ).toFixed(5)
            );
      t.push(s);
    }
    return t;
  }
  euclidean(t) {
    var e = this.individuals[0].length - 1,
      r = this.normalization(),
      s = [];
    for (let e = 0; e < t.length; e++)
      t[e] <= this.minsAndMaxs[e][0]
        ? s.push(0)
        : t[e] >= this.minsAndMaxs[e][1]
        ? s.push(1)
        : s.push(
            (
              (t[e] - this.minsAndMaxs[e][0]) /
              (this.minsAndMaxs[e][1] - this.minsAndMaxs[e][0])
            ).toFixed(5)
          );
    var i = [];
    for (const t of r) {
      var o = t.slice(0, e);
      i.push(euclidean(s, o));
    }
    let n = Math.min(...i),
      a = [];
    for (const t in i) i[t] == n && a.push(this.individuals[t][e]);
    return [...new Set(a)];
  }
  manhattan(t) {
    var e = this.individuals[0].length - 1,
      r = this.normalization(),
      s = [];
    for (let e = 0; e < t.length; e++)
      t[e] <= this.minsAndMaxs[e][0]
        ? s.push(0)
        : t[e] >= this.minsAndMaxs[e][1]
        ? s.push(1)
        : s.push(
            (
              (t[e] - this.minsAndMaxs[e][0]) /
              (this.minsAndMaxs[e][1] - this.minsAndMaxs[e][0])
            ).toFixed(5)
          );
    var i = [];
    for (const t of r) {
      var o = t.slice(0, e);
      i.push(manhattan(s, o));
    }
    let n = Math.min(...i),
      a = [];
    for (const t in i) i[t] == n && a.push(this.individuals[t][e]);
    return [...new Set(a)];
  }
}
class KMeans {
  constructor() {
    this.k = 3;
  }
}
class LinearKMeans extends KMeans {
  constructor() {
    super(), (this.data = []);
  }
  clusterize(t, e, r) {
    let s = [];
    this.data = e;
    for (let i = 0; i < r; i++) {
      let r = [];
      for (let s = 0; s < t; s++) {
        let t = e[Math.floor(Math.random() * e.length)];
        for (; -1 != r.findIndex((e) => e === t); )
          t = e[Math.floor(Math.random() * e.length)];
        r.push(t);
      }
      r = r.sort(function (t, e) {
        return t > e ? 1 : e > t ? -1 : 0;
      });
      let i = [],
        o = [],
        n = 0,
        a = 0,
        h = !0;
      e.forEach((t) => {
        (n = 0),
          (a = 0),
          (h = !0),
          r.forEach((e) => {
            let r = this.distance(t, e);
            h
              ? ((a = Math.abs(r)), (n = e), (h = !h))
              : Math.abs(r) < a && ((a = Math.abs(r)), (n = e)),
              i.push([t, e, r]);
          }),
          o.push([t, n, a]);
      });
      let l = [],
        u = 0,
        c = [];
      do {
        (l = c),
          (c = []),
          (u = 0),
          r.forEach((t, e) => {
            let s = o.filter((e) => e[1] == t).map((t) => t[0]);
            c.push(this.calculateMeanVariance(s)),
              (u += s[1]),
              (r[e] = c[e][0]);
          }),
          Number.isNaN(u) && (u = 0),
          (i = []),
          (o = []),
          (n = 0),
          (a = 0),
          (h = !0),
          e.forEach((t) => {
            (n = 0),
              (a = 0),
              (h = !0),
              r.forEach((e) => {
                let r = this.distance(t, e);
                h
                  ? ((a = Math.abs(r)), (n = e), (h = !h))
                  : Math.abs(r) < a && ((a = Math.abs(r)), (n = e)),
                  i.push([t, e, r]);
              }),
              o.push([t, n, a]);
          });
      } while (JSON.stringify(l) != JSON.stringify(c));
      s.push([u, r, o]);
    }
    return (
      s.sort(function (t, e) {
        return t[0] > e[0] ? 1 : e[0] > t[0] ? -1 : 0;
      }),
      s[0][2]
    );
  }
  distance(t, e) {
    return e - t;
  }
  calculateMeanVariance(t) {
    var e = t.reduce(function (t, e) {
        return t + e;
      }),
      r = (function (t, e) {
        return t.reduce(function (t, r) {
          return t + Math.pow(r - e, 2);
        }, 0);
      })(t, e / t.length),
      s = { mean: e / t.length, variance: r / t.length };
    return [s.mean, s.variance];
  }
}
class _2DKMeans extends KMeans {
  constructor() {
    super(), (this.data = []);
  }
  clusterize(t, e, r) {
    let s = [];
    this.data = e;
    for (let i = 0; i < r; i++) {
      let i = [];
      for (let r = 0; r < t; r++) {
        let t = e[Math.floor(Math.random() * e.length)];
        for (; -1 != i.findIndex((e) => e === t); )
          t = e[Math.floor(Math.random() * e.length)];
        i.push(t);
      }
      let o = [],
        n = [],
        a = [],
        h = 0,
        l = !0;
      e.forEach((t) => {
        (a = [0]),
          (h = 0),
          (l = !0),
          i.forEach((e) => {
            let r = this.distance(t, e);
            l
              ? ((h = Math.abs(r)), (a = e), (l = !l))
              : Math.abs(r) < h && ((h = Math.abs(r)), (a = e)),
              o.push([t, e, r]);
          }),
          n.push([t, a, h]);
      });
      let u = [],
        c = 0,
        f = [],
        m = 0;
      do {
        (u = f),
          (f = []),
          (c = 0),
          i.forEach((t, e) => {
            let r = n
              .filter((e) => JSON.stringify(e[1]) == JSON.stringify(t))
              .map((t) => t[0]);
            f.push([
              this.calculateMeanVariance(r.map((t) => t[0])),
              this.calculateMeanVariance(r.map((t) => t[1])),
            ]),
              (c += r[1]),
              (i[e] = f[e][0]);
          }),
          Number.isNaN(c) && (c = 0),
          (o = []),
          (n = []),
          (a = 0),
          (h = 0),
          (l = !0),
          e.forEach((t) => {
            (a = 0),
              (h = 0),
              (l = !0),
              i.forEach((e) => {
                let r = this.distance(t, e);
                l
                  ? ((h = Math.abs(r)), (a = e), (l = !l))
                  : Math.abs(r) < h && ((h = Math.abs(r)), (a = e)),
                  o.push([t, e, r]);
              }),
              n.push([t, a, h]);
          }),
          m++;
      } while (JSON.stringify(u) != JSON.stringify(f) && m < r);
      s.push([c, i, n]);
    }
    return (
      s.sort(function (t, e) {
        return t[0] > e[0] ? 1 : e[0] > t[0] ? -1 : 0;
      }),
      s[0][2]
    );
  }
  distance(t, e) {
    let r = t[0],
      s = t[1],
      i = e[0],
      o = e[1];
    return Math.sqrt(Math.pow(i - r, 2) + Math.pow(o - s, 2));
  }
  calculateMeanVariance(t) {
    if (t.length < 1) return [1e6, 1e6];
    var e = t.reduce(function (t, e) {
        return t + e;
      }),
      r = (function (t, e) {
        return t.reduce(function (t, r) {
          return t + Math.pow(r - e, 2);
        }, 0);
      })(t, e / t.length),
      s = { mean: e / t.length, variance: r / t.length };
    return [s.mean, s.variance];
  }
}
const distance = (t, e) =>
  Math.sqrt(t.map((t, r) => e[r] - t).reduce((t, e) => t + e * e, 0));
class LinearModel {
  constructor() {
    this.isFit = !1;
  }
}
class LinearRegression extends LinearModel {
  constructor() {
    super(), (this.m = 0), (this.b = 0);
  }
  fit(t, e) {
    for (var r = 0, s = 0, i = 0, o = 0, n = 0; n < t.length; n++)
      (r += t[n]), (s += e[n]), (i += t[n] * e[n]), (o += t[n] * t[n]);
    (this.m =
      (t.length * i - r * s) / (t.length * o - Math.pow(Math.abs(r), 2))),
      (this.b = (s * o - r * i) / (t.length * o - Math.pow(Math.abs(r), 2))),
      (this.isFit = !0);
  }
  predict(t) {
    var e = [];
    if (this.isFit)
      for (var r = 0; r < t.length; r++) e.push(this.m * t[r] + this.b);
    return e;
  }
  mserror(t, e) {
    for (var r = 0, s = 0; s < t.length; s++) r += Math.pow(t[s] - e[s], 2);
    return r / t.length;
  }
  coeficientR2(t, h) {
    for (var r = 0, e = 0, s = 0, i = 0; i < t.length; i++) r += t[i];
    r /= t.length;
    for (i = 0; i < h.length; i++) e += Math.pow(h[i] - r, 2);
    for (i = 0; i < t.length; i++) s += Math.pow(t[i] - r, 2);
    return e / s;
  }
}
class NaiveBayes {
  constructor() {
    this.causes = [];
  }
  insertCause(t, e) {
    let r = !0;
    for (let s = 0; s < this.causes.length; s++) {
      if (this.causes[s][0] === t) {
        (r = !1),
          console.log(
            "Naive Bayes - Error on insertCause: cause names must be unique"
          );
        break;
      }
      if (this.causes[s][1].length != e.length) {
        (r = !1),
          console.log(
            "Naive Bayes - Error on insertCause: all array lengths must be the same"
          );
        break;
      }
    }
    r && this.causes.push([t, e]);
  }
  predict(t, e) {
    let r = this.getCauseByName(t),
      s = [],
      i = [];
    for (let t = 0; t < r.length; t++)
      i.includes(r[t]) || (s.push([r[t], 0]), i.push(r[t]));
    console.log(s);
    for (let r = 0; r < s.length; r++) {
      let i = this.getSimpleProbability([t, s[r][0]]);
      for (let o = 0; o < e.length; o++)
        i *= this.getConditionalProbability(e[o], [t, s[r][0]]);
      s[r][1] = i;
    }
    console.log(s);
    let o = 0,
      n = "nothing :(";
    for (let t = 0; t < s.length; t++)
      s[t][1] > o && ((o = s[t][1]), (n = s[t][0]));
    return [n, 100 * o + "%"];
  }
  getSimpleProbability(t) {
    let e = this.getCauseByName(t[0]),
      r = 0;
    for (let s = 0; s < e.length; s++) e[s] == t[1] && r++;
    return r / e.length;
  }
  getConditionalProbability(t, e) {
    let r = this.getCauseByName(t[0]),
      s = this.getCauseByName(e[0]),
      i = 0,
      o = 0;
    for (let n = 0; n < r.length; n++)
      s[n] == e[1] && (o++, r[n] == t[1] && i++);
    return i / o;
  }
  getCauseByName(t) {
    for (let e = 0; e < this.causes.length; e++)
      if (this.causes[e][0] == t) return this.causes[e][1];
    return null;
  }
}
class Matriz {
  constructor(t, e) {
    (this.rows = t), (this.cols = e), (this.data = []);
    for (let t = 0; t < this.rows; t++) {
      this.data[t] = [];
      for (let e = 0; e < this.cols; e++) this.data[t][e] = 0;
    }
  }
  static multiplicar(t, e) {
    if (t.cols === e.rows) {
      let r = new Matriz(t.rows, e.cols);
      for (let s = 0; s < r.rows; s++)
        for (let i = 0; i < r.cols; i++) {
          let o = 0;
          for (let r = 0; r < t.cols; r++) o += t.data[s][r] * e.data[r][i];
          r.data[s][i] = o;
        }
      return r;
    }
    console.log("Cannot Operate, Check Matriz Multiplication Rules.");
  }
  multiplicar(t) {
    if (t instanceof Matriz)
      for (let e = 0; e < this.rows; e++)
        for (let r = 0; r < this.cols; r++) this.data[e][r] *= t.data[e][r];
    else
      for (let e = 0; e < this.rows; e++)
        for (let r = 0; r < this.cols; r++) this.data[e][r] *= t;
  }
  summar(t) {
    if (t instanceof Matriz)
      for (let e = 0; e < this.rows; e++)
        for (let r = 0; r < this.cols; r++) this.data[e][r] += t.data[e][r];
    else
      for (let e = 0; e < this.rows; e++)
        for (let r = 0; r < this.cols; r++) this.data[e][r] += t;
  }
  static resstar(t, e) {
    let r = new Matriz(t.rows, t.cols);
    for (let s = 0; s < t.rows; s++)
      for (let i = 0; i < t.cols; i++)
        r.data[s][i] = t.data[s][i] - e.data[s][i];
    return r;
  }
  mapear(t) {
    for (let e = 0; e < this.rows; e++)
      for (let r = 0; r < this.cols; r++) {
        let s = this.data[e][r];
        this.data[e][r] = t(s);
      }
  }
  static mapear(t, e) {
    for (let r = 0; r < t.rows; r++)
      for (let s = 0; s < t.cols; s++) {
        let i = t.data[r][s];
        t.data[r][s] = e(i);
      }
    return t;
  }
  tirar_random() {
    for (let t = 0; t < this.rows; t++)
      for (let e = 0; e < this.cols; e++)
        this.data[t][e] = 2 * Math.random() - 1;
  }
  static transpuesta(t) {
    let e = new Matriz(t.cols, t.rows);
    for (let r = 0; r < t.rows; r++)
      for (let s = 0; s < t.cols; s++) e.data[s][r] = t.data[r][s];
    return e;
  }
  imprimir() {
    console.table(this.data);
  }
  convert_to_array() {
    let t = [];
    for (let e = 0; e < this.rows; e++)
      for (let r = 0; r < this.cols; r++) t.push(this.data[e][r]);
    return t;
  }
  static get_array(t) {
    let e = new Matriz(t.length, 1);
    for (let r = 0; r < t.length; r++) e.data[r][0] = t[r];
    return e;
  }
}
class LayerLink {
  constructor(t, e) {
    (this.weights = new Matriz(e, t)),
      (this.bias = new Matriz(e, 1)),
      this.weights.tirar_random(),
      this.bias.tirar_random();
  }
  actualizar_Weights(t) {
    this.weights = t;
  }
  obtener_Weights() {
    return this.weights;
  }
  obtener_Bias() {
    return this.bias;
  }
  summar(t, e) {
    this.weights.summar(t), this.bias.summar(e);
  }
}
class NeuralNetwork {
  constructor(t, e) {
    if (t.length < 2)
      return (
        console.error("Neural Network Needs Atleast 2 Layers To Work."),
        { layers: t }
      );
    if (
      ((this.options = {
        activation: function (t) {
          return 1 / (1 + Math.exp(-t));
        },
        derivative: function (t) {
          return t * (1 - t);
        },
      }),
      (this.learning_rate = 0.1),
      e)
    ) {
      if (
        (e.learning_rate &&
          this.Set_aprendizaje_rate(parseFloat(e.learning_rate)),
        !(
          e.activation &&
          e.derivative &&
          e.activation instanceof Function &&
          e.derivative instanceof Function
        ))
      )
        return (
          console.error(
            "Check Documentation to Learn How To Set Custom Activation Function"
          ),
          console.warn("Documentation Link: http://github.com/AlexDenver"),
          { options: e }
        );
      (this.options.activation = e.activation),
        (this.options.derivative = e.derivative);
    }
    (this.layerCount = t.length - 1),
      (this.inputs = t[0]),
      (this.output_nodes = t[t.length - 1]),
      (this.layerLink = []);
    for (let e = 1, r = 0; r < this.layerCount; e++, r++) {
      if (t[e] <= 0)
        return (
          console.error("A Layer Needs To Have Atleast One Node (Neuron)."),
          { layers: t }
        );
      this.layerLink[r] = new LayerLink(t[r], t[e]);
    }
  }
  Predecir(t) {
    if (t.length !== this.inputs)
      return (
        console.error(
          `This Instance of NeuralNetwork Expects ${this.inputs} Inputs, ${t.length} Provided.`
        ),
        { inputs: t }
      );
    let e = Matriz.get_array(t);
    for (let t = 0; t < this.layerCount; t++)
      (e = Matriz.multiplicar(this.layerLink[t].obtener_Weights(), e)).summar(
        this.layerLink[t].obtener_Bias()
      ),
        e.mapear(this.options.activation);
    return e.convert_to_array();
  }
  Set_aprendizaje_rate(t) {
    t > 1 && t < 100
      ? ((t /= 100) > 0.3 &&
          console.warn("It is recommended to Set Lower Learning Rates"),
        (this.learning_rate = t))
      : console.error("Set Learning Rate Between (0 - 1) or (1 - 100)");
  }
  Entrenar(t, e) {
    if (t.length !== this.inputs)
      return (
        console.error(
          `This Instance of NeuralNetwork Expects ${this.inputs} Inputs, ${t.length} Provided.`
        ),
        { inputs: t }
      );
    if (e.length !== this.output_nodes)
      return (
        console.error(
          `This Instance of NeuralNetwork Expects ${this.output_nodes} Outputs, ${e.length} Provided.`
        ),
        { outputs: e }
      );
    let r = Matriz.get_array(t),
      s = [];
    s[0] = r;
    for (let t = 0; t < this.layerCount; t++)
      (s[t + 1] = Matriz.multiplicar(
        this.layerLink[t].obtener_Weights(),
        s[t]
      )),
        s[t + 1].summar(this.layerLink[t].obtener_Bias()),
        s[t + 1].mapear(this.options.activation);
    let i = Matriz.get_array(e),
      o = [],
      n = [];
    o[this.layerCount] = Matriz.resstar(i, s[this.layerCount]);
    for (let t = this.layerCount; t > 0; t--) {
      (n[t] = Matriz.mapear(s[t], this.options.derivative)),
        n[t].multiplicar(o[t]),
        n[t].multiplicar(this.learning_rate);
      let e = Matriz.transpuesta(s[t - 1]),
        r = Matriz.multiplicar(n[t], e);
      this.layerLink[t - 1].summar(r, n[t]),
        (o[t - 1] = Matriz.multiplicar(
          Matriz.transpuesta(this.layerLink[t - 1].obtener_Weights()),
          o[t]
        ));
    }
  }
}
class PLS {
  constructor(t, e) {
    if (!0 === t)
      (this.meanX = e.meanX),
        (this.stdDevX = e.stdDevX),
        (this.meanY = e.meanY),
        (this.stdDevY = e.stdDevY),
        (this.PBQ = Matrix.checkMatrix(e.PBQ)),
        (this.R2X = e.R2X),
        (this.scale = e.scale),
        (this.scaleMethod = e.scaleMethod),
        (this.tolerance = e.tolerance);
    else {
      let { tolerance: e = 1e-5, scale: r = !0 } = t;
      (this.tolerance = e),
        (this.scale = r),
        (this.latentVectors = t.latentVectors);
    }
  }
  train(t, e) {
    if (
      ((t = Matrix.checkMatrix(t)),
      (e = Matrix.checkMatrix(e)),
      t.length !== e.length)
    )
      throw new RangeError(
        "The number of X rows must be equal to the number of Y rows"
      );
    (this.meanX = t.mean("column")),
      (this.stdDevX = t.standardDeviation("column", {
        mean: this.meanX,
        unbiased: !0,
      })),
      (this.meanY = e.mean("column")),
      (this.stdDevY = e.standardDeviation("column", {
        mean: this.meanY,
        unbiased: !0,
      })),
      this.scale &&
        ((t = t.clone().subRowVector(this.meanX).divRowVector(this.stdDevX)),
        (e = e.clone().subRowVector(this.meanY).divRowVector(this.stdDevY))),
      void 0 === this.latentVectors &&
        (this.latentVectors = Math.min(t.rows - 1, t.columns));
    let r,
      s,
      i,
      o,
      n = t.rows,
      a = t.columns,
      h = e.rows,
      l = e.columns,
      u = t.clone().mul(t).sum(),
      c = e.clone().mul(e).sum(),
      f = this.tolerance,
      m = this.latentVectors,
      p = Matrix.zeros(n, m),
      w = Matrix.zeros(a, m),
      g = Matrix.zeros(h, m),
      d = Matrix.zeros(l, m),
      y = Matrix.zeros(m, m),
      b = w.clone(),
      M = 0;
    for (; norm(e) > f && M < m; ) {
      let a = t.transpose(),
        h = e.transpose(),
        l = maxSumColIndex(t.clone().mul(t)),
        u = maxSumColIndex(e.clone().mul(e)),
        c = t.getColumnVector(l),
        m = e.getColumnVector(u);
      for (r = Matrix.zeros(n, 1); norm(c.clone().sub(r)) > f; )
        (s = a.mmul(m)).div(norm(s)),
          (r = c),
          (c = t.mmul(s)),
          (i = h.mmul(c)).div(norm(i)),
          (m = e.mmul(i));
      r = c;
      let x = a.mmul(r),
        v = r.transpose().mmul(r).get(0, 0),
        k = norm((o = x.div(v)));
      o.div(k),
        r.mul(k),
        s.mul(k),
        (x = m.transpose().mmul(r)),
        (v = r.transpose().mmul(r).get(0, 0));
      let R = x.div(v).get(0, 0);
      t.sub(r.mmul(o.transpose())),
        e.sub(r.clone().mul(R).mmul(i.transpose())),
        p.setColumn(M, r),
        w.setColumn(M, o),
        g.setColumn(M, m),
        d.setColumn(M, i),
        b.setColumn(M, s),
        y.set(M, M, R),
        M++;
    }
    M--,
      (p = p.subMatrix(0, p.rows - 1, 0, M)),
      (w = w.subMatrix(0, w.rows - 1, 0, M)),
      (g = g.subMatrix(0, g.rows - 1, 0, M)),
      (d = d.subMatrix(0, d.rows - 1, 0, M)),
      (b = b.subMatrix(0, b.rows - 1, 0, M)),
      (y = y.subMatrix(0, M, 0, M)),
      (this.ssqYcal = c),
      (this.E = t),
      (this.F = e),
      (this.T = p),
      (this.P = w),
      (this.U = g),
      (this.Q = d),
      (this.W = b),
      (this.B = y),
      (this.PBQ = w.mmul(y).mmul(d.transpose())),
      (this.R2X = r
        .transpose()
        .mmul(r)
        .mmul(o.transpose().mmul(o))
        .div(u)
        .get(0, 0));
  }
  predict(t) {
    let e = Matrix.checkMatrix(t);
    this.scale && (e = e.subRowVector(this.meanX).divRowVector(this.stdDevX));
    let r = e.mmul(this.PBQ);
    return r.mulRowVector(this.stdDevY).addRowVector(this.meanY);
  }
  getExplainedVariance() {
    return this.R2X;
  }
  toJSON() {
    return {
      name: "PLS",
      R2X: this.R2X,
      meanX: this.meanX,
      stdDevX: this.stdDevX,
      meanY: this.meanY,
      stdDevY: this.stdDevY,
      PBQ: this.PBQ,
      tolerance: this.tolerance,
      scale: this.scale,
    };
  }
  static load(t) {
    if ("PLS" !== t.name) throw new RangeError(`Invalid model: ${t.name}`);
    return new PLS(!0, t);
  }
}
function maxSumColIndex(t) {
  return Matrix.rowVector(t.sum("column")).maxIndex()[0];
}
function norm(t) {
  return Math.sqrt(t.clone().apply(pow2array).sum());
}
function pow2array(t, e) {
  this.set(t, e, this.get(t, e) ** 2);
}
function featureNormalize(t) {
  let e = t.mean("column"),
    r = t.standardDeviation("column", { mean: e, unbiased: !0 });
  return {
    result: Matrix.checkMatrix(t).subRowVector(e).divRowVector(r),
    means: e,
    std: r,
  };
}
function initializeMatrices(t, e) {
  if (e)
    for (let e = 0; e < t.length; ++e)
      for (let r = 0; r < t[e].length; ++r) {
        let s = t[e][r];
        t[e][r] = null !== s ? new Matrix(t[e][r]) : void 0;
      }
  else for (let e = 0; e < t.length; ++e) t[e] = new Matrix(t[e]);
  return t;
}
class AbstractMatrix {
  static from1DArray(t, e, r) {
    if (t * e !== r.length)
      throw new RangeError("data length does not match given dimensions");
    let s = new Matrix(t, e);
    for (let i = 0; i < t; i++)
      for (let t = 0; t < e; t++) s.set(i, t, r[i * e + t]);
    return s;
  }
  static rowVector(t) {
    let e = new Matrix(1, t.length);
    for (let r = 0; r < t.length; r++) e.set(0, r, t[r]);
    return e;
  }
  static columnVector(t) {
    let e = new Matrix(t.length, 1);
    for (let r = 0; r < t.length; r++) e.set(r, 0, t[r]);
    return e;
  }
  static zeros(t, e) {
    return new Matrix(t, e);
  }
  static ones(t, e) {
    return new Matrix(t, e).fill(1);
  }
  static rand(t, e, r = {}) {
    if ("object" != typeof r) throw new TypeError("options must be an object");
    const { random: s = Math.random } = r;
    let i = new Matrix(t, e);
    for (let r = 0; r < t; r++) for (let t = 0; t < e; t++) i.set(r, t, s());
    return i;
  }
  static randInt(t, e, r = {}) {
    if ("object" != typeof r) throw new TypeError("options must be an object");
    const { min: s = 0, max: i = 1e3, random: o = Math.random } = r;
    if (!Number.isInteger(s)) throw new TypeError("min must be an integer");
    if (!Number.isInteger(i)) throw new TypeError("max must be an integer");
    if (s >= i) throw new RangeError("min must be smaller than max");
    let n = i - s,
      a = new Matrix(t, e);
    for (let r = 0; r < t; r++)
      for (let t = 0; t < e; t++) {
        let e = s + Math.round(o() * n);
        a.set(r, t, e);
      }
    return a;
  }
  static eye(t, e, r) {
    void 0 === e && (e = t), void 0 === r && (r = 1);
    let s = Math.min(t, e),
      i = this.zeros(t, e);
    for (let t = 0; t < s; t++) i.set(t, t, r);
    return i;
  }
  static diag(t, e, r) {
    let s = t.length;
    void 0 === e && (e = s), void 0 === r && (r = e);
    let i = Math.min(s, e, r),
      o = this.zeros(e, r);
    for (let e = 0; e < i; e++) o.set(e, e, t[e]);
    return o;
  }
  static min(t, e) {
    (t = this.checkMatrix(t)), (e = this.checkMatrix(e));
    let r = t.rows,
      s = t.columns,
      i = new Matrix(r, s);
    for (let o = 0; o < r; o++)
      for (let r = 0; r < s; r++)
        i.set(o, r, Math.min(t.get(o, r), e.get(o, r)));
    return i;
  }
  static max(t, e) {
    (t = this.checkMatrix(t)), (e = this.checkMatrix(e));
    let r = t.rows,
      s = t.columns,
      i = new this(r, s);
    for (let o = 0; o < r; o++)
      for (let r = 0; r < s; r++)
        i.set(o, r, Math.max(t.get(o, r), e.get(o, r)));
    return i;
  }
  static checkMatrix(t) {
    return AbstractMatrix.isMatrix(t) ? t : new Matrix(t);
  }
  static isMatrix(t) {
    return null != t && "Matrix" === t.klass;
  }
  get size() {
    return this.rows * this.columns;
  }
  apply(t) {
    if ("function" != typeof t)
      throw new TypeError("callback must be a function");
    for (let e = 0; e < this.rows; e++)
      for (let r = 0; r < this.columns; r++) t.call(this, e, r);
    return this;
  }
  to1DArray() {
    let t = [];
    for (let e = 0; e < this.rows; e++)
      for (let r = 0; r < this.columns; r++) t.push(this.get(e, r));
    return t;
  }
  to2DArray() {
    let t = [];
    for (let e = 0; e < this.rows; e++) {
      t.push([]);
      for (let r = 0; r < this.columns; r++) t[e].push(this.get(e, r));
    }
    return t;
  }
  toJSON() {
    return this.to2DArray();
  }
  isRowVector() {
    return 1 === this.rows;
  }
  isColumnVector() {
    return 1 === this.columns;
  }
  isVector() {
    return 1 === this.rows || 1 === this.columns;
  }
  isSquare() {
    return this.rows === this.columns;
  }
  isEmpty() {
    return 0 === this.rows || 0 === this.columns;
  }
  isSymmetric() {
    if (this.isSquare()) {
      for (let t = 0; t < this.rows; t++)
        for (let e = 0; e <= t; e++)
          if (this.get(t, e) !== this.get(e, t)) return !1;
      return !0;
    }
    return !1;
  }
  isEchelonForm() {
    let t = 0,
      e = 0,
      r = -1,
      s = !0,
      i = !1;
    for (; t < this.rows && s; ) {
      for (e = 0, i = !1; e < this.columns && !1 === i; )
        0 === this.get(t, e)
          ? e++
          : 1 === this.get(t, e) && e > r
          ? ((i = !0), (r = e))
          : ((s = !1), (i = !0));
      t++;
    }
    return s;
  }
  isReducedEchelonForm() {
    let t = 0,
      e = 0,
      r = -1,
      s = !0,
      i = !1;
    for (; t < this.rows && s; ) {
      for (e = 0, i = !1; e < this.columns && !1 === i; )
        0 === this.get(t, e)
          ? e++
          : 1 === this.get(t, e) && e > r
          ? ((i = !0), (r = e))
          : ((s = !1), (i = !0));
      for (let r = e + 1; r < this.rows; r++) 0 !== this.get(t, r) && (s = !1);
      t++;
    }
    return s;
  }
  echelonForm() {
    let t = this.clone(),
      e = 0,
      r = 0;
    for (; e < t.rows && r < t.columns; ) {
      let s = e;
      for (let i = e; i < t.rows; i++) t.get(i, r) > t.get(s, r) && (s = i);
      if (0 === t.get(s, r)) r++;
      else {
        t.swapRows(e, s);
        let i = t.get(e, r);
        for (let s = r; s < t.columns; s++) t.set(e, s, t.get(e, s) / i);
        for (let s = e + 1; s < t.rows; s++) {
          let i = t.get(s, r) / t.get(e, r);
          t.set(s, r, 0);
          for (let o = r + 1; o < t.columns; o++)
            t.set(s, o, t.get(s, o) - t.get(e, o) * i);
        }
        e++, r++;
      }
    }
    return t;
  }
  reducedEchelonForm() {
    let t = this.echelonForm(),
      e = t.columns,
      r = t.rows,
      s = r - 1;
    for (; s >= 0; )
      if (0 === t.maxRow(s)) s--;
      else {
        let i = 0,
          o = !1;
        for (; i < r && !1 === o; ) 1 === t.get(s, i) ? (o = !0) : i++;
        for (let r = 0; r < s; r++) {
          let o = t.get(r, i);
          for (let n = i; n < e; n++) {
            let e = t.get(r, n) - o * t.get(s, n);
            t.set(r, n, e);
          }
        }
        s--;
      }
    return t;
  }
  set() {
    throw new Error("set method is unimplemented");
  }
  get() {
    throw new Error("get method is unimplemented");
  }
  repeat(t = {}) {
    if ("object" != typeof t) throw new TypeError("options must be an object");
    const { rows: e = 1, columns: r = 1 } = t;
    if (!Number.isInteger(e) || e <= 0)
      throw new TypeError("rows must be a positive integer");
    if (!Number.isInteger(r) || r <= 0)
      throw new TypeError("columns must be a positive integer");
    let s = new Matrix(this.rows * e, this.columns * r);
    for (let t = 0; t < e; t++)
      for (let e = 0; e < r; e++)
        s.setSubMatrix(this, this.rows * t, this.columns * e);
    return s;
  }
  fill(t) {
    for (let e = 0; e < this.rows; e++)
      for (let r = 0; r < this.columns; r++) this.set(e, r, t);
    return this;
  }
  neg() {
    return this.mulS(-1);
  }
  getRow(t) {
    checkRowIndex(this, t);
    let e = [];
    for (let r = 0; r < this.columns; r++) e.push(this.get(t, r));
    return e;
  }
  getRowVector(t) {
    return Matrix.rowVector(this.getRow(t));
  }
  setRow(t, e) {
    checkRowIndex(this, t), (e = checkRowVector(this, e));
    for (let r = 0; r < this.columns; r++) this.set(t, r, e[r]);
    return this;
  }
  swapRows(t, e) {
    checkRowIndex(this, t), checkRowIndex(this, e);
    for (let r = 0; r < this.columns; r++) {
      let s = this.get(t, r);
      this.set(t, r, this.get(e, r)), this.set(e, r, s);
    }
    return this;
  }
  getColumn(t) {
    checkColumnIndex(this, t);
    let e = [];
    for (let r = 0; r < this.rows; r++) e.push(this.get(r, t));
    return e;
  }
  getColumnVector(t) {
    return Matrix.columnVector(this.getColumn(t));
  }
  setColumn(t, e) {
    checkColumnIndex(this, t), (e = checkColumnVector(this, e));
    for (let r = 0; r < this.rows; r++) this.set(r, t, e[r]);
    return this;
  }
  swapColumns(t, e) {
    checkColumnIndex(this, t), checkColumnIndex(this, e);
    for (let r = 0; r < this.rows; r++) {
      let s = this.get(r, t);
      this.set(r, t, this.get(r, e)), this.set(r, e, s);
    }
    return this;
  }
  addRowVector(t) {
    t = checkRowVector(this, t);
    for (let e = 0; e < this.rows; e++)
      for (let r = 0; r < this.columns; r++)
        this.set(e, r, this.get(e, r) + t[r]);
    return this;
  }
  subRowVector(t) {
    t = checkRowVector(this, t);
    for (let e = 0; e < this.rows; e++)
      for (let r = 0; r < this.columns; r++)
        this.set(e, r, this.get(e, r) - t[r]);
    return this;
  }
  mulRowVector(t) {
    t = checkRowVector(this, t);
    for (let e = 0; e < this.rows; e++)
      for (let r = 0; r < this.columns; r++)
        this.set(e, r, this.get(e, r) * t[r]);
    return this;
  }
  divRowVector(t) {
    t = checkRowVector(this, t);
    for (let e = 0; e < this.rows; e++)
      for (let r = 0; r < this.columns; r++)
        this.set(e, r, this.get(e, r) / t[r]);
    return this;
  }
  addColumnVector(t) {
    t = checkColumnVector(this, t);
    for (let e = 0; e < this.rows; e++)
      for (let r = 0; r < this.columns; r++)
        this.set(e, r, this.get(e, r) + t[e]);
    return this;
  }
  subColumnVector(t) {
    t = checkColumnVector(this, t);
    for (let e = 0; e < this.rows; e++)
      for (let r = 0; r < this.columns; r++)
        this.set(e, r, this.get(e, r) - t[e]);
    return this;
  }
  mulColumnVector(t) {
    t = checkColumnVector(this, t);
    for (let e = 0; e < this.rows; e++)
      for (let r = 0; r < this.columns; r++)
        this.set(e, r, this.get(e, r) * t[e]);
    return this;
  }
  divColumnVector(t) {
    t = checkColumnVector(this, t);
    for (let e = 0; e < this.rows; e++)
      for (let r = 0; r < this.columns; r++)
        this.set(e, r, this.get(e, r) / t[e]);
    return this;
  }
  mulRow(t, e) {
    checkRowIndex(this, t);
    for (let r = 0; r < this.columns; r++) this.set(t, r, this.get(t, r) * e);
    return this;
  }
  mulColumn(t, e) {
    checkColumnIndex(this, t);
    for (let r = 0; r < this.rows; r++) this.set(r, t, this.get(r, t) * e);
    return this;
  }
  max() {
    if (this.isEmpty()) return NaN;
    let t = this.get(0, 0);
    for (let e = 0; e < this.rows; e++)
      for (let r = 0; r < this.columns; r++)
        this.get(e, r) > t && (t = this.get(e, r));
    return t;
  }
  maxIndex() {
    checkNonEmpty(this);
    let t = this.get(0, 0),
      e = [0, 0];
    for (let r = 0; r < this.rows; r++)
      for (let s = 0; s < this.columns; s++)
        this.get(r, s) > t && ((t = this.get(r, s)), (e[0] = r), (e[1] = s));
    return e;
  }
  min() {
    if (this.isEmpty()) return NaN;
    let t = this.get(0, 0);
    for (let e = 0; e < this.rows; e++)
      for (let r = 0; r < this.columns; r++)
        this.get(e, r) < t && (t = this.get(e, r));
    return t;
  }
  minIndex() {
    checkNonEmpty(this);
    let t = this.get(0, 0),
      e = [0, 0];
    for (let r = 0; r < this.rows; r++)
      for (let s = 0; s < this.columns; s++)
        this.get(r, s) < t && ((t = this.get(r, s)), (e[0] = r), (e[1] = s));
    return e;
  }
  maxRow(t) {
    if ((checkRowIndex(this, t), this.isEmpty())) return NaN;
    let e = this.get(t, 0);
    for (let r = 1; r < this.columns; r++)
      this.get(t, r) > e && (e = this.get(t, r));
    return e;
  }
  maxRowIndex(t) {
    checkRowIndex(this, t), checkNonEmpty(this);
    let e = this.get(t, 0),
      r = [t, 0];
    for (let s = 1; s < this.columns; s++)
      this.get(t, s) > e && ((e = this.get(t, s)), (r[1] = s));
    return r;
  }
  minRow(t) {
    if ((checkRowIndex(this, t), this.isEmpty())) return NaN;
    let e = this.get(t, 0);
    for (let r = 1; r < this.columns; r++)
      this.get(t, r) < e && (e = this.get(t, r));
    return e;
  }
  minRowIndex(t) {
    checkRowIndex(this, t), checkNonEmpty(this);
    let e = this.get(t, 0),
      r = [t, 0];
    for (let s = 1; s < this.columns; s++)
      this.get(t, s) < e && ((e = this.get(t, s)), (r[1] = s));
    return r;
  }
  maxColumn(t) {
    if ((checkColumnIndex(this, t), this.isEmpty())) return NaN;
    let e = this.get(0, t);
    for (let r = 1; r < this.rows; r++)
      this.get(r, t) > e && (e = this.get(r, t));
    return e;
  }
  maxColumnIndex(t) {
    checkColumnIndex(this, t), checkNonEmpty(this);
    let e = this.get(0, t),
      r = [0, t];
    for (let s = 1; s < this.rows; s++)
      this.get(s, t) > e && ((e = this.get(s, t)), (r[0] = s));
    return r;
  }
  minColumn(t) {
    if ((checkColumnIndex(this, t), this.isEmpty())) return NaN;
    let e = this.get(0, t);
    for (let r = 1; r < this.rows; r++)
      this.get(r, t) < e && (e = this.get(r, t));
    return e;
  }
  minColumnIndex(t) {
    checkColumnIndex(this, t), checkNonEmpty(this);
    let e = this.get(0, t),
      r = [0, t];
    for (let s = 1; s < this.rows; s++)
      this.get(s, t) < e && ((e = this.get(s, t)), (r[0] = s));
    return r;
  }
  diag() {
    let t = Math.min(this.rows, this.columns),
      e = [];
    for (let r = 0; r < t; r++) e.push(this.get(r, r));
    return e;
  }
  norm(t = "frobenius") {
    let e = 0;
    if ("max" === t) return this.max();
    if ("frobenius" === t) {
      for (let t = 0; t < this.rows; t++)
        for (let r = 0; r < this.columns; r++)
          e += this.get(t, r) * this.get(t, r);
      return Math.sqrt(e);
    }
    throw new RangeError(`unknown norm type: ${t}`);
  }
  cumulativeSum() {
    let t = 0;
    for (let e = 0; e < this.rows; e++)
      for (let r = 0; r < this.columns; r++)
        (t += this.get(e, r)), this.set(e, r, t);
    return this;
  }
  dot(t) {
    AbstractMatrix.isMatrix(t) && (t = t.to1DArray());
    let e = this.to1DArray();
    if (e.length !== t.length)
      throw new RangeError("vectors do not have the same size");
    let r = 0;
    for (let s = 0; s < e.length; s++) r += e[s] * t[s];
    return r;
  }
  mmul(t) {
    t = Matrix.checkMatrix(t);
    let e = this.rows,
      r = this.columns,
      s = t.columns,
      i = new Matrix(e, s),
      o = new Float64Array(r);
    for (let n = 0; n < s; n++) {
      for (let e = 0; e < r; e++) o[e] = t.get(e, n);
      for (let t = 0; t < e; t++) {
        let e = 0;
        for (let s = 0; s < r; s++) e += this.get(t, s) * o[s];
        i.set(t, n, e);
      }
    }
    return i;
  }
  strassen2x2(t) {
    t = Matrix.checkMatrix(t);
    let e = new Matrix(2, 2);
    const r = this.get(0, 0),
      s = t.get(0, 0),
      i = this.get(0, 1),
      o = t.get(0, 1),
      n = this.get(1, 0),
      a = t.get(1, 0),
      h = this.get(1, 1),
      l = t.get(1, 1),
      u = (r + h) * (s + l),
      c = (n + h) * s,
      f = r * (o - l),
      m = h * (a - s),
      p = (r + i) * l,
      w = u + m - p + (i - h) * (a + l),
      g = f + p,
      d = c + m,
      y = u - c + f + (n - r) * (s + o);
    return e.set(0, 0, w), e.set(0, 1, g), e.set(1, 0, d), e.set(1, 1, y), e;
  }
  strassen3x3(t) {
    t = Matrix.checkMatrix(t);
    let e = new Matrix(3, 3);
    const r = this.get(0, 0),
      s = this.get(0, 1),
      i = this.get(0, 2),
      o = this.get(1, 0),
      n = this.get(1, 1),
      a = this.get(1, 2),
      h = this.get(2, 0),
      l = this.get(2, 1),
      u = this.get(2, 2),
      c = t.get(0, 0),
      f = t.get(0, 1),
      m = t.get(0, 2),
      p = t.get(1, 0),
      w = t.get(1, 1),
      g = t.get(1, 2),
      d = t.get(2, 0),
      y = t.get(2, 1),
      b = t.get(2, 2),
      M = (r - o) * (-f + w),
      x = (-r + o + n) * (c - f + w),
      v = (o + n) * (-c + f),
      k = r * c,
      R = (-r + h + l) * (c - m + g),
      E = (-r + h) * (m - g),
      A = (h + l) * (-c + m),
      S = (-i + l + u) * (w + d - y),
      C = (i - u) * (w - y),
      N = i * d,
      T = (l + u) * (-d + y),
      I = (-i + n + a) * (g + d - b),
      z = (i - a) * (g - b),
      V = (n + a) * (-d + b),
      D = k + N + s * p,
      j = (r + s + i - o - n - l - u) * w + x + v + k + S + N + T,
      P = k + R + A + (r + s + i - n - a - h - l) * g + N + I + V,
      F = M + n * (-c + f + p - w - g - d + b) + x + k + N + I + z,
      B = M + x + v + k + a * y,
      $ = N + I + z + V + o * m,
      _ = k + R + E + l * (-c + m + p - w - g - d + y) + S + C + N,
      q = S + C + N + T + h * f,
      O = k + R + E + A + u * b;
    return (
      e.set(0, 0, D),
      e.set(0, 1, j),
      e.set(0, 2, P),
      e.set(1, 0, F),
      e.set(1, 1, B),
      e.set(1, 2, $),
      e.set(2, 0, _),
      e.set(2, 1, q),
      e.set(2, 2, O),
      e
    );
  }
  mmulStrassen(t) {
    t = Matrix.checkMatrix(t);
    let e = this.clone(),
      r = e.rows,
      s = e.columns,
      i = t.rows,
      o = t.columns;
    function n(t, e, r) {
      let s = t.rows,
        i = t.columns;
      if (s === e && i === r) return t;
      {
        let s = AbstractMatrix.zeros(e, r);
        return (s = s.setSubMatrix(t, 0, 0));
      }
    }
    s !== i &&
      console.warn(
        `Multiplying ${r} x ${s} and ${i} x ${o} matrix: dimensions do not match.`
      );
    let a = Math.max(r, i),
      h = Math.max(s, o);
    return (function t(e, r, s, i) {
      if (s <= 512 || i <= 512) return e.mmul(r);
      s % 2 == 1 && i % 2 == 1
        ? ((e = n(e, s + 1, i + 1)), (r = n(r, s + 1, i + 1)))
        : s % 2 == 1
        ? ((e = n(e, s + 1, i)), (r = n(r, s + 1, i)))
        : i % 2 == 1 && ((e = n(e, s, i + 1)), (r = n(r, s, i + 1)));
      let o = parseInt(e.rows / 2, 10),
        a = parseInt(e.columns / 2, 10),
        h = e.subMatrix(0, o - 1, 0, a - 1),
        l = r.subMatrix(0, o - 1, 0, a - 1),
        u = e.subMatrix(0, o - 1, a, e.columns - 1),
        c = r.subMatrix(0, o - 1, a, r.columns - 1),
        f = e.subMatrix(o, e.rows - 1, 0, a - 1),
        m = r.subMatrix(o, r.rows - 1, 0, a - 1),
        p = e.subMatrix(o, e.rows - 1, a, e.columns - 1),
        w = r.subMatrix(o, r.rows - 1, a, r.columns - 1),
        g = t(AbstractMatrix.add(h, p), AbstractMatrix.add(l, w), o, a),
        d = t(AbstractMatrix.add(f, p), l, o, a),
        y = t(h, AbstractMatrix.sub(c, w), o, a),
        b = t(p, AbstractMatrix.sub(m, l), o, a),
        M = t(AbstractMatrix.add(h, u), w, o, a),
        x = t(AbstractMatrix.sub(f, h), AbstractMatrix.add(l, c), o, a),
        v = t(AbstractMatrix.sub(u, p), AbstractMatrix.add(m, w), o, a),
        k = AbstractMatrix.add(g, b);
      k.sub(M), k.add(v);
      let R = AbstractMatrix.add(y, M),
        E = AbstractMatrix.add(d, b),
        A = AbstractMatrix.sub(g, d);
      A.add(y), A.add(x);
      let S = AbstractMatrix.zeros(2 * k.rows, 2 * k.columns);
      return (S = (S = (S = (S = S.setSubMatrix(k, 0, 0)).setSubMatrix(
        R,
        k.rows,
        0
      )).setSubMatrix(E, 0, k.columns)).setSubMatrix(
        A,
        k.rows,
        k.columns
      )).subMatrix(0, s - 1, 0, i - 1);
    })((e = n(e, a, h)), (t = n(t, a, h)), a, h);
  }
  scaleRows(t = {}) {
    if ("object" != typeof t) throw new TypeError("options must be an object");
    const { min: e = 0, max: r = 1 } = t;
    if (!Number.isFinite(e)) throw new TypeError("min must be a number");
    if (!Number.isFinite(r)) throw new TypeError("max must be a number");
    if (e >= r) throw new RangeError("min must be smaller than max");
    let s = new Matrix(this.rows, this.columns);
    for (let t = 0; t < this.rows; t++) {
      const i = this.getRow(t);
      i.length > 0 && rescale(i, { min: e, max: r, output: i }), s.setRow(t, i);
    }
    return s;
  }
  scaleColumns(t = {}) {
    if ("object" != typeof t) throw new TypeError("options must be an object");
    const { min: e = 0, max: r = 1 } = t;
    if (!Number.isFinite(e)) throw new TypeError("min must be a number");
    if (!Number.isFinite(r)) throw new TypeError("max must be a number");
    if (e >= r) throw new RangeError("min must be smaller than max");
    let s = new Matrix(this.rows, this.columns);
    for (let t = 0; t < this.columns; t++) {
      const i = this.getColumn(t);
      i.length && rescale(i, { min: e, max: r, output: i }), s.setColumn(t, i);
    }
    return s;
  }
  flipRows() {
    const t = Math.ceil(this.columns / 2);
    for (let e = 0; e < this.rows; e++)
      for (let r = 0; r < t; r++) {
        let t = this.get(e, r),
          s = this.get(e, this.columns - 1 - r);
        this.set(e, r, s), this.set(e, this.columns - 1 - r, t);
      }
    return this;
  }
  flipColumns() {
    const t = Math.ceil(this.rows / 2);
    for (let e = 0; e < this.columns; e++)
      for (let r = 0; r < t; r++) {
        let t = this.get(r, e),
          s = this.get(this.rows - 1 - r, e);
        this.set(r, e, s), this.set(this.rows - 1 - r, e, t);
      }
    return this;
  }
  kroneckerProduct(t) {
    t = Matrix.checkMatrix(t);
    let e = this.rows,
      r = this.columns,
      s = t.rows,
      i = t.columns,
      o = new Matrix(e * s, r * i);
    for (let n = 0; n < e; n++)
      for (let e = 0; e < r; e++)
        for (let r = 0; r < s; r++)
          for (let a = 0; a < i; a++)
            o.set(s * n + r, i * e + a, this.get(n, e) * t.get(r, a));
    return o;
  }
  kroneckerSum(t) {
    if (((t = Matrix.checkMatrix(t)), !this.isSquare() || !t.isSquare()))
      throw new Error("Kronecker Sum needs two Square Matrices");
    let e = this.rows,
      r = t.rows,
      s = this.kroneckerProduct(Matrix.eye(r, r)),
      i = Matrix.eye(e, e).kroneckerProduct(t);
    return s.add(i);
  }
  transpose() {
    let t = new Matrix(this.columns, this.rows);
    for (let e = 0; e < this.rows; e++)
      for (let r = 0; r < this.columns; r++) t.set(r, e, this.get(e, r));
    return t;
  }
  sortRows(t = compareNumbers) {
    for (let e = 0; e < this.rows; e++) this.setRow(e, this.getRow(e).sort(t));
    return this;
  }
  sortColumns(t = compareNumbers) {
    for (let e = 0; e < this.columns; e++)
      this.setColumn(e, this.getColumn(e).sort(t));
    return this;
  }
  subMatrix(t, e, r, s) {
    checkRange(this, t, e, r, s);
    let i = new Matrix(e - t + 1, s - r + 1);
    for (let o = t; o <= e; o++)
      for (let e = r; e <= s; e++) i.set(o - t, e - r, this.get(o, e));
    return i;
  }
  subMatrixRow(t, e, r) {
    if (
      (void 0 === e && (e = 0),
      void 0 === r && (r = this.columns - 1),
      e > r || e < 0 || e >= this.columns || r < 0 || r >= this.columns)
    )
      throw new RangeError("Argument out of range");
    let s = new Matrix(t.length, r - e + 1);
    for (let i = 0; i < t.length; i++)
      for (let o = e; o <= r; o++) {
        if (t[i] < 0 || t[i] >= this.rows)
          throw new RangeError(`Row index out of range: ${t[i]}`);
        s.set(i, o - e, this.get(t[i], o));
      }
    return s;
  }
  subMatrixColumn(t, e, r) {
    if (
      (void 0 === e && (e = 0),
      void 0 === r && (r = this.rows - 1),
      e > r || e < 0 || e >= this.rows || r < 0 || r >= this.rows)
    )
      throw new RangeError("Argument out of range");
    let s = new Matrix(r - e + 1, t.length);
    for (let i = 0; i < t.length; i++)
      for (let o = e; o <= r; o++) {
        if (t[i] < 0 || t[i] >= this.columns)
          throw new RangeError(`Column index out of range: ${t[i]}`);
        s.set(o - e, i, this.get(o, t[i]));
      }
    return s;
  }
  setSubMatrix(t, e, r) {
    if ((t = Matrix.checkMatrix(t)).isEmpty()) return this;
    checkRange(this, e, e + t.rows - 1, r, r + t.columns - 1);
    for (let s = 0; s < t.rows; s++)
      for (let i = 0; i < t.columns; i++) this.set(e + s, r + i, t.get(s, i));
    return this;
  }
  selection(t, e) {
    let r = checkIndices(this, t, e),
      s = new Matrix(t.length, e.length);
    for (let t = 0; t < r.row.length; t++) {
      let e = r.row[t];
      for (let i = 0; i < r.column.length; i++) {
        let o = r.column[i];
        s.set(t, i, this.get(e, o));
      }
    }
    return s;
  }
  trace() {
    let t = Math.min(this.rows, this.columns),
      e = 0;
    for (let r = 0; r < t; r++) e += this.get(r, r);
    return e;
  }
  clone() {
    let t = new Matrix(this.rows, this.columns);
    for (let e = 0; e < this.rows; e++)
      for (let r = 0; r < this.columns; r++) t.set(e, r, this.get(e, r));
    return t;
  }
  sum(t) {
    switch (t) {
      case "row":
        return sumByRow(this);
      case "column":
        return sumByColumn(this);
      case void 0:
        return sumAll(this);
      default:
        throw new Error(`invalid option: ${t}`);
    }
  }
  product(t) {
    switch (t) {
      case "row":
        return productByRow(this);
      case "column":
        return productByColumn(this);
      case void 0:
        return productAll(this);
      default:
        throw new Error(`invalid option: ${t}`);
    }
  }
  mean(t) {
    const e = this.sum(t);
    switch (t) {
      case "row":
        for (let t = 0; t < this.rows; t++) e[t] /= this.columns;
        return e;
      case "column":
        for (let t = 0; t < this.columns; t++) e[t] /= this.rows;
        return e;
      case void 0:
        return e / this.size;
      default:
        throw new Error(`invalid option: ${t}`);
    }
  }
  variance(t, e = {}) {
    if (("object" == typeof t && ((e = t), (t = void 0)), "object" != typeof e))
      throw new TypeError("options must be an object");
    const { unbiased: r = !0, mean: s = this.mean(t) } = e;
    if ("boolean" != typeof r)
      throw new TypeError("unbiased must be a boolean");
    switch (t) {
      case "row":
        if (!Array.isArray(s)) throw new TypeError("mean must be an array");
        return varianceByRow(this, r, s);
      case "column":
        if (!Array.isArray(s)) throw new TypeError("mean must be an array");
        return varianceByColumn(this, r, s);
      case void 0:
        if ("number" != typeof s) throw new TypeError("mean must be a number");
        return varianceAll(this, r, s);
      default:
        throw new Error(`invalid option: ${t}`);
    }
  }
  standardDeviation(t, e) {
    "object" == typeof t && ((e = t), (t = void 0));
    const r = this.variance(t, e);
    if (void 0 === t) return Math.sqrt(r);
    for (let t = 0; t < r.length; t++) r[t] = Math.sqrt(r[t]);
    return r;
  }
  center(t, e = {}) {
    if (("object" == typeof t && ((e = t), (t = void 0)), "object" != typeof e))
      throw new TypeError("options must be an object");
    const { center: r = this.mean(t) } = e;
    switch (t) {
      case "row":
        if (!Array.isArray(r)) throw new TypeError("center must be an array");
        return centerByRow(this, r), this;
      case "column":
        if (!Array.isArray(r)) throw new TypeError("center must be an array");
        return centerByColumn(this, r), this;
      case void 0:
        if ("number" != typeof r)
          throw new TypeError("center must be a number");
        return centerAll(this, r), this;
      default:
        throw new Error(`invalid option: ${t}`);
    }
  }
  scale(t, e = {}) {
    if (("object" == typeof t && ((e = t), (t = void 0)), "object" != typeof e))
      throw new TypeError("options must be an object");
    let r = e.scale;
    switch (t) {
      case "row":
        if (void 0 === r) r = getScaleByRow(this);
        else if (!Array.isArray(r))
          throw new TypeError("scale must be an array");
        return scaleByRow(this, r), this;
      case "column":
        if (void 0 === r) r = getScaleByColumn(this);
        else if (!Array.isArray(r))
          throw new TypeError("scale must be an array");
        return scaleByColumn(this, r), this;
      case void 0:
        if (void 0 === r) r = getScaleAll(this);
        else if ("number" != typeof r)
          throw new TypeError("scale must be a number");
        return scaleAll(this, r), this;
      default:
        throw new Error(`invalid option: ${t}`);
    }
  }
  toString(t) {
    return inspectMatrixWithOptions(this, t);
  }
}
function compareNumbers(t, e) {
  return t - e;
}
(AbstractMatrix.prototype.klass = "Matrix"),
  "undefined" != typeof Symbol &&
    (AbstractMatrix.prototype[Symbol.for("nodejs.util.inspect.custom")] =
      inspectMatrix),
  (AbstractMatrix.random = AbstractMatrix.rand),
  (AbstractMatrix.randomInt = AbstractMatrix.randInt),
  (AbstractMatrix.diagonal = AbstractMatrix.diag),
  (AbstractMatrix.prototype.diagonal = AbstractMatrix.prototype.diag),
  (AbstractMatrix.identity = AbstractMatrix.eye),
  (AbstractMatrix.prototype.negate = AbstractMatrix.prototype.neg),
  (AbstractMatrix.prototype.tensorProduct =
    AbstractMatrix.prototype.kroneckerProduct);
class Matrix extends AbstractMatrix {
  constructor(t, e) {
    if ((super(), Matrix.isMatrix(t))) return t.clone();
    if (Number.isInteger(t) && t >= 0) {
      if (((this.data = []), !(Number.isInteger(e) && e >= 0)))
        throw new TypeError("nColumns must be a positive integer");
      for (let r = 0; r < t; r++) this.data.push(new Float64Array(e));
    } else {
      if (!Array.isArray(t))
        throw new TypeError(
          "First argument must be a positive number or an array"
        );
      {
        const r = t;
        if ("number" != typeof (e = (t = r.length) ? r[0].length : 0))
          throw new TypeError(
            "Data must be a 2D array with at least one element"
          );
        this.data = [];
        for (let s = 0; s < t; s++) {
          if (r[s].length !== e)
            throw new RangeError("Inconsistent array dimensions");
          this.data.push(Float64Array.from(r[s]));
        }
      }
    }
    (this.rows = t), (this.columns = e);
  }
  set(t, e, r) {
    return (this.data[t][e] = r), this;
  }
  get(t, e) {
    return this.data[t][e];
  }
  removeRow(t) {
    return (
      checkRowIndex(this, t), this.data.splice(t, 1), (this.rows -= 1), this
    );
  }
  addRow(t, e) {
    return (
      void 0 === e && ((e = t), (t = this.rows)),
      checkRowIndex(this, t, !0),
      (e = Float64Array.from(checkRowVector(this, e))),
      this.data.splice(t, 0, e),
      (this.rows += 1),
      this
    );
  }
  removeColumn(t) {
    checkColumnIndex(this, t);
    for (let e = 0; e < this.rows; e++) {
      const r = new Float64Array(this.columns - 1);
      for (let s = 0; s < t; s++) r[s] = this.data[e][s];
      for (let s = t + 1; s < this.columns; s++) r[s - 1] = this.data[e][s];
      this.data[e] = r;
    }
    return (this.columns -= 1), this;
  }
  addColumn(t, e) {
    void 0 === e && ((e = t), (t = this.columns)),
      checkColumnIndex(this, t, !0),
      (e = checkColumnVector(this, e));
    for (let r = 0; r < this.rows; r++) {
      const s = new Float64Array(this.columns + 1);
      let i = 0;
      for (; i < t; i++) s[i] = this.data[r][i];
      for (s[i++] = e[r]; i < this.columns + 1; i++) s[i] = this.data[r][i - 1];
      this.data[r] = s;
    }
    return (this.columns += 1), this;
  }
}
installMathOperations(AbstractMatrix, Matrix);
const indent = " ".repeat(2),
  indentData = " ".repeat(4);
function inspectMatrix() {
  return inspectMatrixWithOptions(this);
}
function inspectMatrixWithOptions(t, e = {}) {
  const { maxRows: r = 15, maxColumns: s = 10, maxNumSize: i = 8 } = e;
  return `${t.constructor.name} {\n${indent}[\n${indentData}${inspectData(
    t,
    r,
    s,
    i
  )}\n${indent}]\n${indent}rows: ${t.rows}\n${indent}columns: ${t.columns}\n}`;
}
function inspectData(t, e, r, s) {
  const { rows: i, columns: o } = t,
    n = Math.min(i, e),
    a = Math.min(o, r),
    h = [];
  for (let e = 0; e < n; e++) {
    let r = [];
    for (let i = 0; i < a; i++) r.push(formatNumber(t.get(e, i), s));
    h.push(`${r.join(" ")}`);
  }
  return (
    a !== o && (h[h.length - 1] += ` ... ${o - r} more columns`),
    n !== i && h.push(`... ${i - e} more rows`),
    h.join(`\n${indentData}`)
  );
}
function formatNumber(t, e) {
  const r = String(t);
  if (r.length <= e) return r.padEnd(e, " ");
  const s = t.toPrecision(e - 2);
  if (s.length <= e) return s;
  const i = t.toExponential(e - 2),
    o = i.indexOf("e"),
    n = i.slice(o);
  return i.slice(0, e - n.length) + n;
}
function installMathOperations(t, e) {
  (t.prototype.add = function (t) {
    return "number" == typeof t ? this.addS(t) : this.addM(t);
  }),
    (t.prototype.addS = function (t) {
      for (let e = 0; e < this.rows; e++)
        for (let r = 0; r < this.columns; r++)
          this.set(e, r, this.get(e, r) + t);
      return this;
    }),
    (t.prototype.addM = function (t) {
      if (
        ((t = e.checkMatrix(t)),
        this.rows !== t.rows || this.columns !== t.columns)
      )
        throw new RangeError("Matrices dimensions must be equal");
      for (let e = 0; e < this.rows; e++)
        for (let r = 0; r < this.columns; r++)
          this.set(e, r, this.get(e, r) + t.get(e, r));
      return this;
    }),
    (t.add = function (t, r) {
      return new e(t).add(r);
    }),
    (t.prototype.sub = function (t) {
      return "number" == typeof t ? this.subS(t) : this.subM(t);
    }),
    (t.prototype.subS = function (t) {
      for (let e = 0; e < this.rows; e++)
        for (let r = 0; r < this.columns; r++)
          this.set(e, r, this.get(e, r) - t);
      return this;
    }),
    (t.prototype.subM = function (t) {
      if (
        ((t = e.checkMatrix(t)),
        this.rows !== t.rows || this.columns !== t.columns)
      )
        throw new RangeError("Matrices dimensions must be equal");
      for (let e = 0; e < this.rows; e++)
        for (let r = 0; r < this.columns; r++)
          this.set(e, r, this.get(e, r) - t.get(e, r));
      return this;
    }),
    (t.sub = function (t, r) {
      return new e(t).sub(r);
    }),
    (t.prototype.subtract = t.prototype.sub),
    (t.prototype.subtractS = t.prototype.subS),
    (t.prototype.subtractM = t.prototype.subM),
    (t.subtract = t.sub),
    (t.prototype.mul = function (t) {
      return "number" == typeof t ? this.mulS(t) : this.mulM(t);
    }),
    (t.prototype.mulS = function (t) {
      for (let e = 0; e < this.rows; e++)
        for (let r = 0; r < this.columns; r++)
          this.set(e, r, this.get(e, r) * t);
      return this;
    }),
    (t.prototype.mulM = function (t) {
      if (
        ((t = e.checkMatrix(t)),
        this.rows !== t.rows || this.columns !== t.columns)
      )
        throw new RangeError("Matrices dimensions must be equal");
      for (let e = 0; e < this.rows; e++)
        for (let r = 0; r < this.columns; r++)
          this.set(e, r, this.get(e, r) * t.get(e, r));
      return this;
    }),
    (t.mul = function (t, r) {
      return new e(t).mul(r);
    }),
    (t.prototype.multiply = t.prototype.mul),
    (t.prototype.multiplyS = t.prototype.mulS),
    (t.prototype.multiplyM = t.prototype.mulM),
    (t.multiply = t.mul),
    (t.prototype.div = function (t) {
      return "number" == typeof t ? this.divS(t) : this.divM(t);
    }),
    (t.prototype.divS = function (t) {
      for (let e = 0; e < this.rows; e++)
        for (let r = 0; r < this.columns; r++)
          this.set(e, r, this.get(e, r) / t);
      return this;
    }),
    (t.prototype.divM = function (t) {
      if (
        ((t = e.checkMatrix(t)),
        this.rows !== t.rows || this.columns !== t.columns)
      )
        throw new RangeError("Matrices dimensions must be equal");
      for (let e = 0; e < this.rows; e++)
        for (let r = 0; r < this.columns; r++)
          this.set(e, r, this.get(e, r) / t.get(e, r));
      return this;
    }),
    (t.div = function (t, r) {
      return new e(t).div(r);
    }),
    (t.prototype.divide = t.prototype.div),
    (t.prototype.divideS = t.prototype.divS),
    (t.prototype.divideM = t.prototype.divM),
    (t.divide = t.div),
    (t.prototype.mod = function (t) {
      return "number" == typeof t ? this.modS(t) : this.modM(t);
    }),
    (t.prototype.modS = function (t) {
      for (let e = 0; e < this.rows; e++)
        for (let r = 0; r < this.columns; r++)
          this.set(e, r, this.get(e, r) % t);
      return this;
    }),
    (t.prototype.modM = function (t) {
      if (
        ((t = e.checkMatrix(t)),
        this.rows !== t.rows || this.columns !== t.columns)
      )
        throw new RangeError("Matrices dimensions must be equal");
      for (let e = 0; e < this.rows; e++)
        for (let r = 0; r < this.columns; r++)
          this.set(e, r, this.get(e, r) % t.get(e, r));
      return this;
    }),
    (t.mod = function (t, r) {
      return new e(t).mod(r);
    }),
    (t.prototype.modulus = t.prototype.mod),
    (t.prototype.modulusS = t.prototype.modS),
    (t.prototype.modulusM = t.prototype.modM),
    (t.modulus = t.mod),
    (t.prototype.and = function (t) {
      return "number" == typeof t ? this.andS(t) : this.andM(t);
    }),
    (t.prototype.andS = function (t) {
      for (let e = 0; e < this.rows; e++)
        for (let r = 0; r < this.columns; r++)
          this.set(e, r, this.get(e, r) & t);
      return this;
    }),
    (t.prototype.andM = function (t) {
      if (
        ((t = e.checkMatrix(t)),
        this.rows !== t.rows || this.columns !== t.columns)
      )
        throw new RangeError("Matrices dimensions must be equal");
      for (let e = 0; e < this.rows; e++)
        for (let r = 0; r < this.columns; r++)
          this.set(e, r, this.get(e, r) & t.get(e, r));
      return this;
    }),
    (t.and = function (t, r) {
      return new e(t).and(r);
    }),
    (t.prototype.or = function (t) {
      return "number" == typeof t ? this.orS(t) : this.orM(t);
    }),
    (t.prototype.orS = function (t) {
      for (let e = 0; e < this.rows; e++)
        for (let r = 0; r < this.columns; r++)
          this.set(e, r, this.get(e, r) | t);
      return this;
    }),
    (t.prototype.orM = function (t) {
      if (
        ((t = e.checkMatrix(t)),
        this.rows !== t.rows || this.columns !== t.columns)
      )
        throw new RangeError("Matrices dimensions must be equal");
      for (let e = 0; e < this.rows; e++)
        for (let r = 0; r < this.columns; r++)
          this.set(e, r, this.get(e, r) | t.get(e, r));
      return this;
    }),
    (t.or = function (t, r) {
      return new e(t).or(r);
    }),
    (t.prototype.xor = function (t) {
      return "number" == typeof t ? this.xorS(t) : this.xorM(t);
    }),
    (t.prototype.xorS = function (t) {
      for (let e = 0; e < this.rows; e++)
        for (let r = 0; r < this.columns; r++)
          this.set(e, r, this.get(e, r) ^ t);
      return this;
    }),
    (t.prototype.xorM = function (t) {
      if (
        ((t = e.checkMatrix(t)),
        this.rows !== t.rows || this.columns !== t.columns)
      )
        throw new RangeError("Matrices dimensions must be equal");
      for (let e = 0; e < this.rows; e++)
        for (let r = 0; r < this.columns; r++)
          this.set(e, r, this.get(e, r) ^ t.get(e, r));
      return this;
    }),
    (t.xor = function (t, r) {
      return new e(t).xor(r);
    }),
    (t.prototype.leftShift = function (t) {
      return "number" == typeof t ? this.leftShiftS(t) : this.leftShiftM(t);
    }),
    (t.prototype.leftShiftS = function (t) {
      for (let e = 0; e < this.rows; e++)
        for (let r = 0; r < this.columns; r++)
          this.set(e, r, this.get(e, r) << t);
      return this;
    }),
    (t.prototype.leftShiftM = function (t) {
      if (
        ((t = e.checkMatrix(t)),
        this.rows !== t.rows || this.columns !== t.columns)
      )
        throw new RangeError("Matrices dimensions must be equal");
      for (let e = 0; e < this.rows; e++)
        for (let r = 0; r < this.columns; r++)
          this.set(e, r, this.get(e, r) << t.get(e, r));
      return this;
    }),
    (t.leftShift = function (t, r) {
      return new e(t).leftShift(r);
    }),
    (t.prototype.signPropagatingRightShift = function (t) {
      return "number" == typeof t
        ? this.signPropagatingRightShiftS(t)
        : this.signPropagatingRightShiftM(t);
    }),
    (t.prototype.signPropagatingRightShiftS = function (t) {
      for (let e = 0; e < this.rows; e++)
        for (let r = 0; r < this.columns; r++)
          this.set(e, r, this.get(e, r) >> t);
      return this;
    }),
    (t.prototype.signPropagatingRightShiftM = function (t) {
      if (
        ((t = e.checkMatrix(t)),
        this.rows !== t.rows || this.columns !== t.columns)
      )
        throw new RangeError("Matrices dimensions must be equal");
      for (let e = 0; e < this.rows; e++)
        for (let r = 0; r < this.columns; r++)
          this.set(e, r, this.get(e, r) >> t.get(e, r));
      return this;
    }),
    (t.signPropagatingRightShift = function (t, r) {
      return new e(t).signPropagatingRightShift(r);
    }),
    (t.prototype.rightShift = function (t) {
      return "number" == typeof t ? this.rightShiftS(t) : this.rightShiftM(t);
    }),
    (t.prototype.rightShiftS = function (t) {
      for (let e = 0; e < this.rows; e++)
        for (let r = 0; r < this.columns; r++)
          this.set(e, r, this.get(e, r) >>> t);
      return this;
    }),
    (t.prototype.rightShiftM = function (t) {
      if (
        ((t = e.checkMatrix(t)),
        this.rows !== t.rows || this.columns !== t.columns)
      )
        throw new RangeError("Matrices dimensions must be equal");
      for (let e = 0; e < this.rows; e++)
        for (let r = 0; r < this.columns; r++)
          this.set(e, r, this.get(e, r) >>> t.get(e, r));
      return this;
    }),
    (t.rightShift = function (t, r) {
      return new e(t).rightShift(r);
    }),
    (t.prototype.zeroFillRightShift = t.prototype.rightShift),
    (t.prototype.zeroFillRightShiftS = t.prototype.rightShiftS),
    (t.prototype.zeroFillRightShiftM = t.prototype.rightShiftM),
    (t.zeroFillRightShift = t.rightShift),
    (t.prototype.not = function () {
      for (let t = 0; t < this.rows; t++)
        for (let e = 0; e < this.columns; e++) this.set(t, e, ~this.get(t, e));
      return this;
    }),
    (t.not = function (t) {
      return new e(t).not();
    }),
    (t.prototype.abs = function () {
      for (let t = 0; t < this.rows; t++)
        for (let e = 0; e < this.columns; e++)
          this.set(t, e, Math.abs(this.get(t, e)));
      return this;
    }),
    (t.abs = function (t) {
      return new e(t).abs();
    }),
    (t.prototype.acos = function () {
      for (let t = 0; t < this.rows; t++)
        for (let e = 0; e < this.columns; e++)
          this.set(t, e, Math.acos(this.get(t, e)));
      return this;
    }),
    (t.acos = function (t) {
      return new e(t).acos();
    }),
    (t.prototype.acosh = function () {
      for (let t = 0; t < this.rows; t++)
        for (let e = 0; e < this.columns; e++)
          this.set(t, e, Math.acosh(this.get(t, e)));
      return this;
    }),
    (t.acosh = function (t) {
      return new e(t).acosh();
    }),
    (t.prototype.asin = function () {
      for (let t = 0; t < this.rows; t++)
        for (let e = 0; e < this.columns; e++)
          this.set(t, e, Math.asin(this.get(t, e)));
      return this;
    }),
    (t.asin = function (t) {
      return new e(t).asin();
    }),
    (t.prototype.asinh = function () {
      for (let t = 0; t < this.rows; t++)
        for (let e = 0; e < this.columns; e++)
          this.set(t, e, Math.asinh(this.get(t, e)));
      return this;
    }),
    (t.asinh = function (t) {
      return new e(t).asinh();
    }),
    (t.prototype.atan = function () {
      for (let t = 0; t < this.rows; t++)
        for (let e = 0; e < this.columns; e++)
          this.set(t, e, Math.atan(this.get(t, e)));
      return this;
    }),
    (t.atan = function (t) {
      return new e(t).atan();
    }),
    (t.prototype.atanh = function () {
      for (let t = 0; t < this.rows; t++)
        for (let e = 0; e < this.columns; e++)
          this.set(t, e, Math.atanh(this.get(t, e)));
      return this;
    }),
    (t.atanh = function (t) {
      return new e(t).atanh();
    }),
    (t.prototype.cbrt = function () {
      for (let t = 0; t < this.rows; t++)
        for (let e = 0; e < this.columns; e++)
          this.set(t, e, Math.cbrt(this.get(t, e)));
      return this;
    }),
    (t.cbrt = function (t) {
      return new e(t).cbrt();
    }),
    (t.prototype.ceil = function () {
      for (let t = 0; t < this.rows; t++)
        for (let e = 0; e < this.columns; e++)
          this.set(t, e, Math.ceil(this.get(t, e)));
      return this;
    }),
    (t.ceil = function (t) {
      return new e(t).ceil();
    }),
    (t.prototype.clz32 = function () {
      for (let t = 0; t < this.rows; t++)
        for (let e = 0; e < this.columns; e++)
          this.set(t, e, Math.clz32(this.get(t, e)));
      return this;
    }),
    (t.clz32 = function (t) {
      return new e(t).clz32();
    }),
    (t.prototype.cos = function () {
      for (let t = 0; t < this.rows; t++)
        for (let e = 0; e < this.columns; e++)
          this.set(t, e, Math.cos(this.get(t, e)));
      return this;
    }),
    (t.cos = function (t) {
      return new e(t).cos();
    }),
    (t.prototype.cosh = function () {
      for (let t = 0; t < this.rows; t++)
        for (let e = 0; e < this.columns; e++)
          this.set(t, e, Math.cosh(this.get(t, e)));
      return this;
    }),
    (t.cosh = function (t) {
      return new e(t).cosh();
    }),
    (t.prototype.exp = function () {
      for (let t = 0; t < this.rows; t++)
        for (let e = 0; e < this.columns; e++)
          this.set(t, e, Math.exp(this.get(t, e)));
      return this;
    }),
    (t.exp = function (t) {
      return new e(t).exp();
    }),
    (t.prototype.expm1 = function () {
      for (let t = 0; t < this.rows; t++)
        for (let e = 0; e < this.columns; e++)
          this.set(t, e, Math.expm1(this.get(t, e)));
      return this;
    }),
    (t.expm1 = function (t) {
      return new e(t).expm1();
    }),
    (t.prototype.floor = function () {
      for (let t = 0; t < this.rows; t++)
        for (let e = 0; e < this.columns; e++)
          this.set(t, e, Math.floor(this.get(t, e)));
      return this;
    }),
    (t.floor = function (t) {
      return new e(t).floor();
    }),
    (t.prototype.fround = function () {
      for (let t = 0; t < this.rows; t++)
        for (let e = 0; e < this.columns; e++)
          this.set(t, e, Math.fround(this.get(t, e)));
      return this;
    }),
    (t.fround = function (t) {
      return new e(t).fround();
    }),
    (t.prototype.log = function () {
      for (let t = 0; t < this.rows; t++)
        for (let e = 0; e < this.columns; e++)
          this.set(t, e, Math.log(this.get(t, e)));
      return this;
    }),
    (t.log = function (t) {
      return new e(t).log();
    }),
    (t.prototype.log1p = function () {
      for (let t = 0; t < this.rows; t++)
        for (let e = 0; e < this.columns; e++)
          this.set(t, e, Math.log1p(this.get(t, e)));
      return this;
    }),
    (t.log1p = function (t) {
      return new e(t).log1p();
    }),
    (t.prototype.log10 = function () {
      for (let t = 0; t < this.rows; t++)
        for (let e = 0; e < this.columns; e++)
          this.set(t, e, Math.log10(this.get(t, e)));
      return this;
    }),
    (t.log10 = function (t) {
      return new e(t).log10();
    }),
    (t.prototype.log2 = function () {
      for (let t = 0; t < this.rows; t++)
        for (let e = 0; e < this.columns; e++)
          this.set(t, e, Math.log2(this.get(t, e)));
      return this;
    }),
    (t.log2 = function (t) {
      return new e(t).log2();
    }),
    (t.prototype.round = function () {
      for (let t = 0; t < this.rows; t++)
        for (let e = 0; e < this.columns; e++)
          this.set(t, e, Math.round(this.get(t, e)));
      return this;
    }),
    (t.round = function (t) {
      return new e(t).round();
    }),
    (t.prototype.sign = function () {
      for (let t = 0; t < this.rows; t++)
        for (let e = 0; e < this.columns; e++)
          this.set(t, e, Math.sign(this.get(t, e)));
      return this;
    }),
    (t.sign = function (t) {
      return new e(t).sign();
    }),
    (t.prototype.sin = function () {
      for (let t = 0; t < this.rows; t++)
        for (let e = 0; e < this.columns; e++)
          this.set(t, e, Math.sin(this.get(t, e)));
      return this;
    }),
    (t.sin = function (t) {
      return new e(t).sin();
    }),
    (t.prototype.sinh = function () {
      for (let t = 0; t < this.rows; t++)
        for (let e = 0; e < this.columns; e++)
          this.set(t, e, Math.sinh(this.get(t, e)));
      return this;
    }),
    (t.sinh = function (t) {
      return new e(t).sinh();
    }),
    (t.prototype.sqrt = function () {
      for (let t = 0; t < this.rows; t++)
        for (let e = 0; e < this.columns; e++)
          this.set(t, e, Math.sqrt(this.get(t, e)));
      return this;
    }),
    (t.sqrt = function (t) {
      return new e(t).sqrt();
    }),
    (t.prototype.tan = function () {
      for (let t = 0; t < this.rows; t++)
        for (let e = 0; e < this.columns; e++)
          this.set(t, e, Math.tan(this.get(t, e)));
      return this;
    }),
    (t.tan = function (t) {
      return new e(t).tan();
    }),
    (t.prototype.tanh = function () {
      for (let t = 0; t < this.rows; t++)
        for (let e = 0; e < this.columns; e++)
          this.set(t, e, Math.tanh(this.get(t, e)));
      return this;
    }),
    (t.tanh = function (t) {
      return new e(t).tanh();
    }),
    (t.prototype.trunc = function () {
      for (let t = 0; t < this.rows; t++)
        for (let e = 0; e < this.columns; e++)
          this.set(t, e, Math.trunc(this.get(t, e)));
      return this;
    }),
    (t.trunc = function (t) {
      return new e(t).trunc();
    }),
    (t.pow = function (t, r) {
      return new e(t).pow(r);
    }),
    (t.prototype.pow = function (t) {
      return "number" == typeof t ? this.powS(t) : this.powM(t);
    }),
    (t.prototype.powS = function (t) {
      for (let e = 0; e < this.rows; e++)
        for (let r = 0; r < this.columns; r++)
          this.set(e, r, Math.pow(this.get(e, r), t));
      return this;
    }),
    (t.prototype.powM = function (t) {
      if (
        ((t = e.checkMatrix(t)),
        this.rows !== t.rows || this.columns !== t.columns)
      )
        throw new RangeError("Matrices dimensions must be equal");
      for (let e = 0; e < this.rows; e++)
        for (let r = 0; r < this.columns; r++)
          this.set(e, r, Math.pow(this.get(e, r), t.get(e, r)));
      return this;
    });
}
function sumByRow(t) {
  let e = newArray(t.rows);
  for (let r = 0; r < t.rows; ++r)
    for (let s = 0; s < t.columns; ++s) e[r] += t.get(r, s);
  return e;
}
function sumByColumn(t) {
  let e = newArray(t.columns);
  for (let r = 0; r < t.rows; ++r)
    for (let s = 0; s < t.columns; ++s) e[s] += t.get(r, s);
  return e;
}
function sumAll(t) {
  let e = 0;
  for (let r = 0; r < t.rows; r++)
    for (let s = 0; s < t.columns; s++) e += t.get(r, s);
  return e;
}
function productByRow(t) {
  let e = newArray(t.rows, 1);
  for (let r = 0; r < t.rows; ++r)
    for (let s = 0; s < t.columns; ++s) e[r] *= t.get(r, s);
  return e;
}
function productByColumn(t) {
  let e = newArray(t.columns, 1);
  for (let r = 0; r < t.rows; ++r)
    for (let s = 0; s < t.columns; ++s) e[s] *= t.get(r, s);
  return e;
}
function productAll(t) {
  let e = 1;
  for (let r = 0; r < t.rows; r++)
    for (let s = 0; s < t.columns; s++) e *= t.get(r, s);
  return e;
}
function varianceByRow(t, e, r) {
  const s = t.rows,
    i = t.columns,
    o = [];
  for (let n = 0; n < s; n++) {
    let s = 0,
      a = 0,
      h = 0;
    for (let e = 0; e < i; e++) (s += h = t.get(n, e) - r[n]), (a += h * h);
    e ? o.push((a - (s * s) / i) / (i - 1)) : o.push((a - (s * s) / i) / i);
  }
  return o;
}
function varianceByColumn(t, e, r) {
  const s = t.rows,
    i = t.columns,
    o = [];
  for (let n = 0; n < i; n++) {
    let i = 0,
      a = 0,
      h = 0;
    for (let e = 0; e < s; e++) (i += h = t.get(e, n) - r[n]), (a += h * h);
    e ? o.push((a - (i * i) / s) / (s - 1)) : o.push((a - (i * i) / s) / s);
  }
  return o;
}
function varianceAll(t, e, r) {
  const s = t.rows,
    i = t.columns,
    o = s * i;
  let n = 0,
    a = 0,
    h = 0;
  for (let e = 0; e < s; e++)
    for (let s = 0; s < i; s++) (n += h = t.get(e, s) - r), (a += h * h);
  return e ? (a - (n * n) / o) / (o - 1) : (a - (n * n) / o) / o;
}
function centerByRow(t, e) {
  for (let r = 0; r < t.rows; r++)
    for (let s = 0; s < t.columns; s++) t.set(r, s, t.get(r, s) - e[r]);
}
function centerByColumn(t, e) {
  for (let r = 0; r < t.rows; r++)
    for (let s = 0; s < t.columns; s++) t.set(r, s, t.get(r, s) - e[s]);
}
function centerAll(t, e) {
  for (let r = 0; r < t.rows; r++)
    for (let s = 0; s < t.columns; s++) t.set(r, s, t.get(r, s) - e);
}
function getScaleByRow(t) {
  const e = [];
  for (let r = 0; r < t.rows; r++) {
    let s = 0;
    for (let e = 0; e < t.columns; e++)
      s += Math.pow(t.get(r, e), 2) / (t.columns - 1);
    e.push(Math.sqrt(s));
  }
  return e;
}
function scaleByRow(t, e) {
  for (let r = 0; r < t.rows; r++)
    for (let s = 0; s < t.columns; s++) t.set(r, s, t.get(r, s) / e[r]);
}
function getScaleByColumn(t) {
  const e = [];
  for (let r = 0; r < t.columns; r++) {
    let s = 0;
    for (let e = 0; e < t.rows; e++)
      s += Math.pow(t.get(e, r), 2) / (t.rows - 1);
    e.push(Math.sqrt(s));
  }
  return e;
}
function scaleByColumn(t, e) {
  for (let r = 0; r < t.rows; r++)
    for (let s = 0; s < t.columns; s++) t.set(r, s, t.get(r, s) / e[s]);
}
function getScaleAll(t) {
  const e = t.size - 1;
  let r = 0;
  for (let s = 0; s < t.columns; s++)
    for (let i = 0; i < t.rows; i++) r += Math.pow(t.get(i, s), 2) / e;
  return Math.sqrt(r);
}
function scaleAll(t, e) {
  for (let r = 0; r < t.rows; r++)
    for (let s = 0; s < t.columns; s++) t.set(r, s, t.get(r, s) / e);
}
function checkRowIndex(t, e, r) {
  let s = r ? t.rows : t.rows - 1;
  if (e < 0 || e > s) throw new RangeError("Row index out of range");
}
function checkColumnIndex(t, e, r) {
  let s = r ? t.columns : t.columns - 1;
  if (e < 0 || e > s) throw new RangeError("Column index out of range");
}
function checkRowVector(t, e) {
  if ((e.to1DArray && (e = e.to1DArray()), e.length !== t.columns))
    throw new RangeError(
      "vector size must be the same as the number of columns"
    );
  return e;
}
function checkColumnVector(t, e) {
  if ((e.to1DArray && (e = e.to1DArray()), e.length !== t.rows))
    throw new RangeError("vector size must be the same as the number of rows");
  return e;
}
function checkIndices(t, e, r) {
  return { row: checkRowIndices(t, e), column: checkColumnIndices(t, r) };
}
function checkRowIndices(t, e) {
  if ("object" != typeof e)
    throw new TypeError("unexpected type for row indices");
  if (e.some((e) => e < 0 || e >= t.rows))
    throw new RangeError("row indices are out of range");
  return Array.isArray(e) || (e = Array.from(e)), e;
}
function checkColumnIndices(t, e) {
  if ("object" != typeof e)
    throw new TypeError("unexpected type for column indices");
  if (e.some((e) => e < 0 || e >= t.columns))
    throw new RangeError("column indices are out of range");
  return Array.isArray(e) || (e = Array.from(e)), e;
}
function checkRange(t, e, r, s, i) {
  if (5 !== arguments.length) throw new RangeError("expected 4 arguments");
  if (
    (checkNumber("startRow", e),
    checkNumber("endRow", r),
    checkNumber("startColumn", s),
    checkNumber("endColumn", i),
    e > r ||
      s > i ||
      e < 0 ||
      e >= t.rows ||
      r < 0 ||
      r >= t.rows ||
      s < 0 ||
      s >= t.columns ||
      i < 0 ||
      i >= t.columns)
  )
    throw new RangeError("Submatrix indices are out of range");
}
function newArray(t, e = 0) {
  let r = [];
  for (let s = 0; s < t; s++) r.push(e);
  return r;
}
function checkNumber(t, e) {
  if ("number" != typeof e) throw new TypeError(`${t} must be a number`);
}
function checkNonEmpty(t) {
  if (t.isEmpty()) throw new Error("Empty matrix has no elements to index");
}
class PolynomialModel {
  constructor() {
    this.isFit = !1;
  }
}
class PolynomialRegression extends PolynomialModel {
  constructor() {
    super(), (this.solutions = []), (this.error = 0);
  }
  fit(t, e, r) {
    let s = r + 1,
      i = r + 2,
      o = new Array(s);
    for (let t = 0; t < s; t++) o[t] = new Array(i);
    for (let r = 0; r < s; r++)
      for (let s = 0; s < i; s++) {
        let n = 0;
        if (0 == r && 0 == s) n = t.length;
        else if (s == i - 1)
          for (let s = 0; s < t.length; s++) n += Math.pow(t[s], r) * e[s];
        else for (let e = 0; e < t.length; e++) n += Math.pow(t[e], s + r);
        o[r][s] = n;
      }
    for (let t = 1; t < s; t++)
      for (let e = 0; e <= t - 1; e++) {
        let r = o[t][e] / o[e][e];
        for (let s = e; s < i; s++) o[t][s] = o[t][s] - r * o[e][s];
      }
    for (let t = s - 1; t > -1; t--)
      for (let e = s - 1; e > -1; e--)
        t == e
          ? (o[t][i - 1] = o[t][i - 1] / o[t][e])
          : 0 != o[t][e] && (o[t][i - 1] -= o[t][e] * o[e][i - 1]);
    this.solutions = new Array(s);
    for (let t = 0; t < s; t++) this.solutions[t] = o[t][i - 1];
    (this.isFit = !0), this.calculateR2(t, e);
  }
  predict(t) {
    let e = [];
    if (this.isFit)
      for (let r = 0; r < t.length; r++) {
        let s = 0;
        for (let e = 0; e < this.solutions.length; e++)
          s += this.solutions[e] * Math.pow(t[r], e);
        e.push(s);
      }
    return e;
  }
  calculateR2(t, e) {
    let r = new Array(t.length),
      s = this.predict(t),
      i = 0;
    for (let o = 0; o < t.length; o++)
      (i += e[o]), (r[o] = Math.pow(e[o] - s[o], 2));
    let o = 0,
      n = 0;
    for (let s = 0; s < t.length; s++)
      (o += r[s]), (n += Math.pow(e[s] - i / t.length, 2));
    let a = (n - o) / n;
    this.error = a;
  }
  getError() {
    return this.error;
  }
}
var svmjs = (function (t) {
  var e = function (t) {};
  function r(t) {
    return function (e, r) {
      for (var s = 0, i = 0; i < e.length; i++)
        s += (e[i] - r[i]) * (e[i] - r[i]);
      return Math.exp(-s / (2 * t * t));
    };
  }
  function s(t, e) {
    for (var r = 0, s = 0; s < t.length; s++) r += t[s] * e[s];
    return r;
  }
  return (
    (e.prototype = {
      train: function (t, e, i) {
        (this.data = t), (this.labels = e);
        var o = (i = i || {}).C || 1,
          n = i.tol || 1e-4,
          a = i.alphatol || 1e-7,
          h = i.maxiter || 1e4,
          l = i.numpasses || 10,
          u = s;
        if (((this.kernelType = "linear"), "kernel" in i))
          if ("string" == typeof i.kernel) {
            if (
              ("linear" === i.kernel && ((this.kernelType = "linear"), (u = s)),
              "rbf" === i.kernel)
            ) {
              var c = i.rbfsigma || 0.5;
              (this.rbfSigma = c), (this.kernelType = "rbf"), (u = r(c));
            }
          } else (this.kernelType = "custom"), (u = i.kernel);
        (this.kernel = u), (this.N = t.length);
        var f = this.N;
        if (
          ((this.D = t[0].length),
          this.D,
          (this.alpha = (function (t) {
            for (var e = new Array(t), r = 0; r < t; r++) e[r] = 0;
            return e;
          })(f)),
          (this.b = 0),
          (this.usew_ = !1),
          i.memoize)
        ) {
          this.kernelResults = new Array(f);
          for (var m = 0; m < f; m++) {
            this.kernelResults[m] = new Array(f);
            for (var p = 0; p < f; p++)
              this.kernelResults[m][p] = u(t[m], t[p]);
          }
        }
        for (var w, g = 0, d = 0; d < l && g < h; ) {
          var y = 0;
          for (m = 0; m < f; m++) {
            var b = this.marginOne(t[m]) - e[m];
            if (
              (e[m] * b < -n && this.alpha[m] < o) ||
              (e[m] * b > n && this.alpha[m] > 0)
            ) {
              for (p = m; p === m; )
                0, (w = this.N), (p = Math.floor(Math.random() * (w - 0) + 0));
              var M = this.marginOne(t[p]) - e[p];
              (ai = this.alpha[m]), (aj = this.alpha[p]);
              var x = 0,
                v = o;
              if (
                (e[m] === e[p]
                  ? ((x = Math.max(0, ai + aj - o)), (v = Math.min(o, ai + aj)))
                  : ((x = Math.max(0, aj - ai)),
                    (v = Math.min(o, o + aj - ai))),
                Math.abs(x - v) < 1e-4)
              )
                continue;
              var k =
                2 * this.kernelResult(m, p) -
                this.kernelResult(m, m) -
                this.kernelResult(p, p);
              if (k >= 0) continue;
              var R = aj - (e[p] * (b - M)) / k;
              if ((R > v && (R = v), R < x && (R = x), Math.abs(aj - R) < 1e-4))
                continue;
              this.alpha[p] = R;
              var E = ai + e[m] * e[p] * (aj - R);
              this.alpha[m] = E;
              var A =
                  this.b -
                  b -
                  e[m] * (E - ai) * this.kernelResult(m, m) -
                  e[p] * (R - aj) * this.kernelResult(m, p),
                S =
                  this.b -
                  M -
                  e[m] * (E - ai) * this.kernelResult(m, p) -
                  e[p] * (R - aj) * this.kernelResult(p, p);
              (this.b = 0.5 * (A + S)),
                E > 0 && E < o && (this.b = A),
                R > 0 && R < o && (this.b = S),
                y++;
            }
          }
          g++, 0 == y ? d++ : (d = 0);
        }
        if ("linear" === this.kernelType)
          for (this.w = new Array(this.D), p = 0; p < this.D; p++) {
            var C = 0;
            for (m = 0; m < this.N; m++) C += this.alpha[m] * e[m] * t[m][p];
            (this.w[p] = C), (this.usew_ = !0);
          }
        else {
          var N = [],
            T = [],
            I = [];
          for (m = 0; m < this.N; m++)
            this.alpha[m] > a &&
              (N.push(this.data[m]),
              T.push(this.labels[m]),
              I.push(this.alpha[m]));
          (this.data = N),
            (this.labels = T),
            (this.alpha = I),
            (this.N = this.data.length);
        }
        var z = {};
        return (z.iters = g), z;
      },
      marginOne: function (t) {
        var e = this.b;
        if (this.usew_) for (var r = 0; r < this.D; r++) e += t[r] * this.w[r];
        else
          for (var s = 0; s < this.N; s++)
            e += this.alpha[s] * this.labels[s] * this.kernel(t, this.data[s]);
        return e;
      },
      predictOne: function (t) {
        return this.marginOne(t) > 0 ? 1 : -1;
      },
      margins: function (t) {
        for (var e = t.length, r = new Array(e), s = 0; s < e; s++)
          r[s] = this.marginOne(t[s]);
        return r;
      },
      kernelResult: function (t, e) {
        return this.kernelResults
          ? this.kernelResults[t][e]
          : this.kernel(this.data[t], this.data[e]);
      },
      predict: function (t) {
        for (var e = this.margins(t), r = 0; r < e.length; r++)
          e[r] = e[r] > 0 ? 1 : -1;
        return e;
      },
      getWeights: function () {
        for (var t = new Array(this.D), e = 0; e < this.D; e++) {
          for (var r = 0, s = 0; s < this.N; s++)
            r += this.alpha[s] * this.labels[s] * this.data[s][e];
          t[e] = r;
        }
        return { w: t, b: this.b };
      },
      toJSON: function () {
        return "custom" === this.kernelType
          ? (console.log(
              "Can't save this SVM because it's using custom, unsupported kernel..."
            ),
            {})
          : ((json = {}),
            (json.N = this.N),
            (json.D = this.D),
            (json.b = this.b),
            (json.kernelType = this.kernelType),
            "linear" === this.kernelType && (json.w = this.w),
            "rbf" === this.kernelType &&
              ((json.rbfSigma = this.rbfSigma),
              (json.data = this.data),
              (json.labels = this.labels),
              (json.alpha = this.alpha)),
            json);
      },
      fromJSON: function (t) {
        (this.N = t.N),
          (this.D = t.D),
          (this.b = t.b),
          (this.kernelType = t.kernelType),
          "linear" === this.kernelType
            ? ((this.w = t.w), (this.usew_ = !0), (this.kernel = s))
            : "rbf" == this.kernelType
            ? ((this.rbfSigma = t.rbfSigma),
              (this.kernel = r(this.rbfSigma)),
              (this.data = t.data),
              (this.labels = t.labels),
              (this.alpha = t.alpha))
            : console.log("ERROR! unrecognized kernel type." + this.kernelType);
      },
    }),
    ((t = t || {}).SVM = e),
    (t.makeRbfKernel = r),
    (t.linearKernel = s),
    t
  );
})("undefined" != typeof module && module.exports);
function joinArrays() {
  var t = [];
  if (6 == arguments.length) {
    t.push([arguments[0], arguments[2], arguments[4]]);
    for (var e = 0; e < arguments[1].length; e++)
      t.push([arguments[1][e], arguments[3][e], arguments[5][e]]);
  }
  return t;
}
function zip(t) {
  return t[0].map(function (e, r) {
    return t.map(function (t) {
      return t[r];
    });
  });
}
function euclidean(t, e) {
  for (var r = 0, s = 0; s < t.length; s++) r += Math.pow(t[s] - e[s], 2);
  return Math.sqrt(r).toFixed(5);
}
function manhattan(t, e) {
  for (var r = 0, s = 0; s < t.length; s++) r += Math.abs(t[s] - e[s]);
  return r;
}
function Perceptron(t) {
  t || (t = {});
  var e,
    r = "debug" in t && t.debug,
    s = "weights" in t ? t.weights.slice() : [],
    i = "threshold" in t ? t.threshold : 1;
  e = "learningrate" in t ? t.learningrate : 0.1;
  var o = [],
    n = {
      weights: s,
      retrain: function () {
        for (var t = o.length, e = !0, r = 0; r < t; r++) {
          var s = o.shift();
          e = n.train(s.input, s.target) && e;
        }
        return e;
      },
      train: function (e, a) {
        for (; s.length < e.length; ) s.push(Math.random());
        s.length == e.length && s.push("bias" in t ? t.bias : 1);
        var h = n.perceive(e);
        if (
          (o.push({ input: e, target: a, prev: h }),
          r && console.log("> training %s, expecting: %s got: %s", e, a, h),
          h == a)
        )
          return !0;
        r && console.log("> adjusting weights...", s, e);
        for (var l = 0; l < s.length; l++) {
          var u = l == e.length ? i : e[l];
          n.adjust(h, a, u, l);
        }
        return r && console.log(" -> weights:", s), !1;
      },
      adjust: function (t, r, i, o) {
        var a = n.delta(t, r, i, e);
        if (((s[o] += a), isNaN(s[o])))
          throw new Error("weights[" + o + "] went to NaN!!");
      },
      delta: function (t, e, r, s) {
        return (e - t) * s * r;
      },
      perceive: function (t, e, r) {
        for (var o = 0, n = 0; n < t.length; n++) o += t[n] * s[n];
        return (
          (o += i * s[s.length - 1]),
          (r = r || ((t) => Number(this.sigmoid(t) >= 0.5)))
            ? r(o)
            : e
            ? o
            : o > 0
            ? 1
            : 0
        );
      },
      sigmoid: function (t) {
        return 1 / (1 + Math.pow(Math.E, -t));
      },
      hardside: function (t) {
        return t;
      },
    };
  return n;
}
class LogisticModel {
  constructor() {}
}
class LogisticRegression extends LogisticModel {
  constructor() {
    super(), (this.alpha = 0.001), (this.lambda = 0), (this.iterations = 100);
  }
  fit(t) {
    console.log(t), (this.dim = t[0].length);
    for (var s = t.length, h = [], i = [], r = 0; r < s; ++r) {
      var a = t[r],
        o = [],
        e = a[a.length - 1];
      o.push(1);
      for (var l = 0; l < a.length - 1; ++l) o.push(a[l]);
      h.push(o), i.push(e);
    }
    this.theta = [];
    for (var n = 0; n < this.dim; ++n) this.theta.push(0);
    for (var c = 0; c < this.iterations; ++c) {
      var g = this.grad(h, i, this.theta);
      for (n = 0; n < this.dim; ++n)
        this.theta[n] = this.theta[n] - this.alpha * g[n];
    }
    return (
      (this.threshold = this.computeThreshold(h, i)),
      {
        theta: this.theta,
        threshold: this.threshold,
        cost: this.cost(h, i, this.theta),
        config: {
          alpha: this.alpha,
          lambda: this.lambda,
          iterations: this.iterations,
        },
      }
    );
  }
  computeThreshold(t, s) {
    for (var h = 1, i = t.length, r = 0; r < i; ++r) {
      var a = this.transform(t[r]);
      1 == s[r] && h > a && (h = a);
    }
    return h;
  }
  grad(t, s, h) {
    try {
      for (var i = t.length, r = [], a = 0; a < this.dim; ++a) {
        for (var o = 0, e = 0; e < i; ++e) {
          var l = t[e];
          o += ((this.h(l, h) - s[e]) * l[a] + this.lambda * h[a]) / i;
        }
        r.push(o);
      }
      return r;
    } catch (t) {
      console.log(t);
    }
  }
  h(t, s) {
    try {
      for (var h = 0, i = 0; i < this.dim; ++i) h += s[i] * t[i];
      return 1 / (1 + Math.exp(-h));
    } catch (t) {
      console.log(t);
    }
  }
  transform(t) {
    if ((console.log("VALOR DE X: " + t), t[0].length)) {
      var s = [];
      console.log("LOGITUD : " + t.length);
      for (var h = 0; h < t.length; ++h) {
        var i = this.transform(t[h]);
        s.push(i);
      }
      return s;
    }
    var r = [];
    r.push(1);
    for (var a = 0; a < t.length; ++a) r.push(t[a]);
    return this.h(r, this.theta);
  }
  cost(t, s, h) {
    for (var i = t.length, r = 0, a = 0; a < i; ++a) {
      var o = s[a],
        e = t[a];
      r +=
        -(o * Math.log(this.h(e, h)) + (1 - o) * Math.log(1 - this.h(e, h))) /
        i;
    }
    for (var l = 0; l < this.dim; ++l)
      r += (this.lambda * h[l] * h[l]) / (2 * i);
    return r;
  }
}
class MultiClassLogistic extends LogisticModel {
  constructor() {
    super(), (this.alpha = 0.001), (this.lambda = 0), (this.iterations = 100);
  }
  fit(t, s) {
    this.dim = t[0].length;
    var h = t.length;
    if (!s) {
      s = [];
      for (var i = 0; i < h; ++i) {
        for (var r = !1, a = t[i][this.dim - 1], o = 0; o < s.length; ++o)
          if (a == s[o]) {
            r = !0;
            break;
          }
        r || s.push(a);
      }
    }
    (this.classes = s), console.log(this.classes), (this.logistics = {});
    for (var e = {}, l = 0; l < this.classes.length; ++l) {
      var n = this.classes[l];
      this.logistics[n] = new LogisticRegression({
        alpha: this.alpha,
        lambda: this.lambda,
        iterations: this.iterations,
      });
      var c = [];
      for (i = 0; i < h; ++i) {
        var g = [];
        for (o = 0; o < this.dim - 1; ++o) g.push(t[i][o]);
        g.push(t[i][this.dim - 1] == n ? 1 : 0), c.push(g);
      }
      e[n] = this.logistics[n].fit(c);
    }
    return e;
  }
  transform(t) {
    if (t[0].length) {
      for (var s = [], h = 0; h < t.length; ++h) {
        var i = this.transform(t[h]);
        s.push(i);
      }
      return s;
    }
    for (var r = 0, a = "", o = 0; o < this.classes.length; ++o) {
      var e = this.classes[o],
        l = this.logistics[e].transform(t);
      r < l && ((r = l), (a = e));
    }
    return a;
  }
}
function calcularDistanciaManhattan(t, l) {
  let a = [],
    e = 0;
  for (; e < t.length; ) {
    let n = 0,
      c = 0,
      r = t[e];
    for (; n < r.length; ) {
      let t = l[1][n],
        a = r[1][n] - t;
      (c += Math.abs(a)), n++;
    }
    a.push(c), e++;
  }
  return console.log(a), a;
}
function calcularDistanciaEuclidiana(t, l) {
  let a = [],
    e = 0;
  for (; e < t.length; ) {
    let n = 0,
      c = 0,
      r = t[e];
    for (; n < r.length; ) {
      let t = l[1][n],
        a = r[1][n] - t;
      (c += Math.pow(a, 2)), n++;
    }
    let i = Math.sqrt(c);
    a.push(i), e++;
  }
  return console.log(a), a;
}
function predecirCluster(t, l, a) {
  let e, n;
  e =
    1 === t
      ? calcularDistanciaManhattan(l, a)
      : calcularDistanciaEuclidiana(l, a);
  for (let t = 0; t < e.length; t++) n ? e[t] < n && (n = e[t]) : (n = e[t]);
  let c = "";
  for (let t = 0; t < e.length; t++)
    e[t] === n && (0 === c.length ? (c = l[t][2]) : (c += "," + l[t][2]));
  return `El individuo ${a[0]} pertenece a: ${c}`;
}
var atributos = new Array(
    new Array("Outlook", "Temperature", "Humidity", "Windy", "Class"),
    new Array("sunny", "hot", "high", "false", "N"),
    new Array("sunny", "hot", "high", "true", "N"),
    new Array("overcast", "hot", "high", "false", "P"),
    new Array("rain", "mild", "high", "false", "P"),
    new Array("rain", "cool", "normal", "false", "P"),
    new Array("rain", "cool", "normal", "true", "N"),
    new Array("overcast", "cool", "normal", "true", "P"),
    new Array("sunny", "mild", "high", "false", "N"),
    new Array("sunny", "cool", "normal", "false", "P"),
    new Array("rain", "mild", "normal", "false", "P"),
    new Array("sunny", "mild", "normal", "true", "P"),
    new Array("overcast", "mild", "high", "true", "P"),
    new Array("overcast", "hot", "normal", "false", "P"),
    new Array("rain", "mild", "high", "true", "N")
  ),
  condicion = new Array("sunny", "hot", "high", "false"),
  varGanar = "P",
  varPerder = "N",
  totalGanar = 0,
  totalPerder = 0,
  total = 0;
function cantidad(e) {
  let t = 0;
  for (let a = 1; a < atributos.length; a++)
    e == atributos[a][atributos[0].length - 1] && t++;
  return t;
}
function probabilidad(e) {
  let t = new Array(),
    a = new Array(),
    r = new Array();
  for (let r = 0; r < condicion.length; r++) {
    let n = condicion[r],
      o = 0,
      l = 0;
    for (let t = 0; t < atributos.length; t++)
      for (let a = 0; a < atributos[t].length; a++)
        n == atributos[t][a] &&
          (e == atributos[t][atributos[0].length - 1] && o++, l++);
    (t[r] = o), (a[r] = l);
  }
  let n = totalPerder;
  e == varGanar && (n = totalGanar);
  for (let e = 0; e < a.length; e++) {
    let o = t[e],
      l = a[e],
      d = ((o / l) * (l / total)) / (n / total);
    r[e] = d;
  }
  return r;
}
function ganar() {
  let e = probabilidad(varGanar),
    t = totalGanar / total;
  for (let a = 0; a < e.length; a++) t *= e[a];
  return t;
}
function perder() {
  let e = probabilidad(varPerder),
    t = totalPerder / total;
  for (let a = 0; a < e.length; a++) t *= e[a];
  return t;
}
function imprimirTabla(e, t) {
  var a = document.createElement("table");
  (a.style.border = "groove"), (a.id = "tab123");
  var r = 0;
  for (let e = 0; e < atributos.length; e++) {
    var n = document.createElement("tr");
    (l = document.createElement("td")).style.border = "groove";
    var o = document.createTextNode("");
    0 != r && (o = document.createTextNode(r)),
      r++,
      l.appendChild(o),
      n.appendChild(l);
    for (let t = 0; t < atributos[0].length; t++) {
      var l;
      (l = document.createElement("td")).style.border = "groove";
      o = document.createTextNode(atributos[e][t]);
      l.appendChild(o), n.appendChild(l);
    }
    a.appendChild(n);
  }
  document.getElementById("salida").appendChild(a);
  var d = document.createTextNode(e),
    i = document.createTextNode(t);
  document.getElementById("salida2").appendChild(d),
    document.getElementById("salida3").appendChild(i);
}
function inicio() {
  (totalGanar = cantidad(varGanar)),
    (totalPerder = cantidad(varPerder)),
    (total = totalGanar + totalPerder);
  let e = ganar(),
    t = perder();
  imprimirTabla(e, t);
  var a = document.createTextNode(""),
    r = document.createTextNode(condicion);
  e > t
    ? (console.log("Ganar"), (a = document.createTextNode("Ganar")))
    : (console.log("No ganar"), (a = document.createTextNode("No ganar"))),
    document.getElementById("salida4").appendChild(a),
    document.getElementById("salida1").appendChild(r);
}
function cambioCondicion() {
  var e = prompt("Escrite la condicion separadas por coma y sin espacio", "");
  condicion = e;
  var t = document.getElementById("tab123");
  document.getElementById("salida").removeChild(t), inicio();
}
inicio();
