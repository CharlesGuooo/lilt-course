/* 冻结 LiLT 上的轻量 head · 共用脚本
   目录与翻页、主题切换、Python 代码高亮、提示框、数据图、交互组件。
   数据来自 data.js（window.HEADS），由 code/ 里的脚本在 FUNSD 上实测后导出；没有 data.js 时依赖它的组件不初始化。 */
(function () {
  'use strict';

  var D = window.HEADS || null;

  var CHAPTERS = [
    { f: 'index.html', n: '··', title: '课程首页' },
    { f: 'ch00_problem.html', n: '00', title: '问题与证据规则' },
    { f: 'ch01_testbed.html', n: '01', title: '实验台：冻结特征与 FUNSD' },
    { f: 'ch02_linear.html', n: '02', title: '基线：线性层与小 MLP' },
    { f: 'ch03_sequence.html', n: '03', title: '序列上下文：卷积 / LSTM / 注意力' },
    { f: 'ch04_crf.html', n: '04', title: '解码层：CRF' },
    { f: 'ch05_layout.html', n: '05', title: '版面感知 head' },
    { f: 'ch06_span.html', n: '06', title: '片段与 biaffine' },
    { f: 'ch07_tree.html', n: '07', title: '树模型 head' },
    { f: 'ch08_prototype.html', n: '08', title: '原型与相似度' },
    { f: 'ch09_compare.html', n: '09', title: '横向对比与验证方案' }
  ];
  var page = document.body.getAttribute('data-page') || 'index.html';
  var cur = -1;
  CHAPTERS.forEach(function (c, i) { if (c.f === page) cur = i; });

  var LAB = ['other', 'question', 'answer', 'header'];
  var LAB_COLOR = ['var(--lab-o)', 'var(--lab-q)', 'var(--lab-a)', 'var(--lab-h)'];
  var LAB_CLS = ['', 'q', 'a', 'h'];

  // ---------------------------------------------------------------- 小工具
  var SVGNS = 'http://www.w3.org/2000/svg';
  function h(tag, attrs, html) {
    var e = document.createElement(tag);
    if (attrs) for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (html != null) e.innerHTML = html;
    return e;
  }
  function s(tag, attrs, parent) {
    var e = document.createElementNS(SVGNS, tag);
    if (attrs) for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function txt(parent, x, y, str, attrs) {
    var a = attrs || {};
    a.x = x; a.y = y;
    var t = s('text', a, parent);
    t.textContent = str;
    return t;
  }
  function esc(t) {
    return String(t).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }
  function lin(d0, d1, r0, r1) {
    var f = function (v) { return r0 + (v - d0) / (d1 - d0) * (r1 - r0); };
    f.inv = function (p) { return d0 + (p - r0) / (r1 - r0) * (d1 - d0); };
    return f;
  }
  function logScale(d0, d1, r0, r1) {
    var a = Math.log(d0), b = Math.log(d1);
    var f = function (v) { return r0 + (Math.log(v) - a) / (b - a) * (r1 - r0); };
    f.inv = function (p) { return Math.exp(a + (p - r0) / (r1 - r0) * (b - a)); };
    return f;
  }
  function fmt(v, d) { return Number(v).toFixed(d == null ? 3 : d); }
  // 按估算宽度截断标签：中日韩字符约 12px，其余约 6.6px（12px 字号）
  function clip(str, px) {
    var w = 0, out = '';
    for (var i = 0; i < str.length; i++) {
      var cw = str.charCodeAt(i) > 0x2e80 ? 12 : 6.6;
      if (w + cw > px - 8 && i < str.length - 1) return out + '…';
      w += cw; out += str[i];
    }
    return out;
  }
  function bytes(b) {
    if (b >= 1e9) return (b / 1e9).toFixed(2) + ' GB';
    if (b >= 1e6) return (b / 1e6).toFixed(1) + ' MB';
    return (b / 1e3).toFixed(1) + ' KB';
  }
  function ms(v) { return v < 1 ? v.toFixed(2) + ' ms' : v < 10 ? v.toFixed(1) + ' ms' : v.toFixed(0) + ' ms'; }

  // ---------------------------------------------------------------- 主题
  var THEMES = [null, 'light', 'dark'];
  var THEME_NAMES = { 'null': '跟随系统', light: '浅色', dark: '深色' };
  function getTheme() { return document.documentElement.getAttribute('data-theme'); }
  function setTheme(t) {
    if (t) document.documentElement.setAttribute('data-theme', t);
    else document.documentElement.removeAttribute('data-theme');
    try {
      if (t) localStorage.setItem('lilt-theme', t); else localStorage.removeItem('lilt-theme');
    } catch (e) { /* 无痕模式等情况下存不了，忽略 */ }
  }

  // ---------------------------------------------------------------- 目录、顶栏、翻页
  function buildNav() {
    var side = document.getElementById('sidebar');
    if (!side) return;
    var items = CHAPTERS.map(function (c, i) {
      return '<li><a href="' + c.f + '"' + (i === cur ? ' class="on" aria-current="page"' : '') +
        '><span class="num">' + c.n + '</span><span>' + c.title + '</span></a></li>';
    }).join('');
    side.innerHTML =
      '<a class="brand" href="index.html"><span class="brand-mark" aria-hidden="true"><i></i><i></i></span>' +
      '<span><b>冻结 LiLT 上的 head</b><small>8 类 head 的实测对照</small></span></a>' +
      '<nav aria-label="章节目录"><ul class="toc">' + items + '</ul></nav>' +
      '<div class="side-foot">' +
      LAB.map(function (l, i) {
        return '<span class="legend-inline"><i style="background:' + LAB_COLOR[i] + '"></i>' + l + '</span>';
      }).join('') +
      '<br><a href="../../html/index.html" style="font-size:13.5px">← 前置课：LiLT 架构课</a><br>' +
      '<button class="theme-btn" type="button"></button></div>';
    var btn = side.querySelector('.theme-btn');
    function label() { btn.textContent = '主题：' + THEME_NAMES[String(getTheme())]; }
    label();
    btn.addEventListener('click', function () {
      var i = THEMES.indexOf(getTheme());
      setTheme(THEMES[(i + 1) % THEMES.length]);
      label();
    });

    var c = CHAPTERS[cur] || CHAPTERS[0];
    var bar = h('div', { 'class': 'topbar' },
      '<button type="button" aria-label="打开章节目录">☰ 目录</button><span>' +
      (cur > 0 ? '第 ' + c.n + ' 章 · ' : '') + c.title + '</span>');
    document.body.insertBefore(bar, document.body.firstChild);
    bar.querySelector('button').addEventListener('click', function (e) {
      e.stopPropagation();
      document.body.classList.toggle('nav-open');
    });
    document.addEventListener('click', function (e) {
      if (document.body.classList.contains('nav-open') && !side.contains(e.target)) {
        document.body.classList.remove('nav-open');
      }
    });
  }

  function buildPager() {
    var el = document.getElementById('pager');
    if (!el || cur < 0) return;
    var prev = CHAPTERS[cur - 1], next = CHAPTERS[cur + 1], html = '';
    if (prev) html += '<a class="prev" href="' + prev.f + '"><small>← 上一章</small>' +
      (prev.n !== '··' ? prev.n + ' · ' : '') + prev.title + '</a>';
    if (next) html += '<a class="next" href="' + next.f + '"><small>下一章 →</small>' + next.n + ' · ' + next.title + '</a>';
    el.innerHTML = html;
  }

  // ---------------------------------------------------------------- Python 高亮
  var PY_RE = new RegExp([
    '(#[^\\n]*)',
    '("""[\\s\\S]*?"""|\'\'\'[\\s\\S]*?\'\'\'|\\b[rbfu]{0,2}"(?:[^"\\\\\\n]|\\\\.)*"|\\b[rbfu]{0,2}\'(?:[^\'\\\\\\n]|\\\\.)*\'|"(?:[^"\\\\\\n]|\\\\.)*"|\'(?:[^\'\\\\\\n]|\\\\.)*\')',
    '\\b(def|return|for|in|if|elif|else|import|from|with|as|lambda|not|and|or|is|None|True|False|class|while|try|except|raise|pass|assert|super)\\b',
    '\\b(\\d+(?:\\.\\d+)?(?:e[-+]?\\d+)?)\\b'
  ].join('|'), 'g');
  function highlightPy(src) {
    var out = '', last = 0, m;
    PY_RE.lastIndex = 0;
    while ((m = PY_RE.exec(src))) {
      out += esc(src.slice(last, m.index));
      var cls = m[1] ? 'com' : m[2] ? 'str' : m[3] ? 'kw' : 'num';
      out += '<span class="' + cls + '">' + esc(m[0]) + '</span>';
      last = PY_RE.lastIndex;
    }
    return out + esc(src.slice(last));
  }
  function highlightAll() {
    var list = document.querySelectorAll('pre.code > code');
    for (var i = 0; i < list.length; i++) {
      if (!list[i].classList.contains('plain')) list[i].innerHTML = highlightPy(list[i].textContent);
    }
  }

  // ---------------------------------------------------------------- 提示框
  var tipEl = null;
  function showTip(html, ev) {
    if (!tipEl) {
      tipEl = h('div', { 'class': 'tip', role: 'status' });
      document.body.appendChild(tipEl);
    }
    tipEl.innerHTML = html;
    tipEl.classList.add('show');
    var r = tipEl.getBoundingClientRect();
    var x = ev.clientX + 14, y = ev.clientY + 14;
    if (x + r.width > window.innerWidth - 8) x = ev.clientX - r.width - 14;
    if (y + r.height > window.innerHeight - 8) y = ev.clientY - r.height - 14;
    tipEl.style.left = Math.max(8, x) + 'px';
    tipEl.style.top = Math.max(8, y) + 'px';
  }
  function hideTip() { if (tipEl) tipEl.classList.remove('show'); }

  // ---------------------------------------------------------------- 图表公共部分
  var renders = [];
  function responsive(fn) { renders.push(fn); fn(); }
  var rt = null;
  window.addEventListener('resize', function () {
    clearTimeout(rt);
    rt = setTimeout(function () { renders.forEach(function (fn) { fn(); }); }, 120);
  });

  function plotHost(el) {
    var host = el.querySelector('.plot');
    host.innerHTML = '';
    return host;
  }
  function legend(el, items) {
    var old = el.querySelector('.legend');
    if (old) old.parentNode.removeChild(old);
    var lg = h('div', { 'class': 'legend' }, items.map(function (it) {
      return '<span><i class="' + (it.sq ? 'sq' : '') + '" style="background:' + it.color + '"></i>' + it.name + '</span>';
    }).join(''));
    el.insertBefore(lg, el.querySelector('.plot'));
  }
  function dataTable(chartEl, title, headers, rows) {
    var d = h('details', { 'class': 'data-table' });
    d.innerHTML = '<summary>' + title + '</summary><div class="table-wrap"><table><thead><tr>' +
      headers.map(function (x, i) { return '<th' + (i > 0 ? ' class="n"' : '') + '>' + x + '</th>'; }).join('') +
      '</tr></thead><tbody></tbody></table></div>';
    var tb = d.querySelector('tbody');
    function set(rs) {
      tb.innerHTML = rs.map(function (r) {
        return '<tr>' + r.map(function (c, i) { return '<td' + (i > 0 ? ' class="n"' : '') + '>' + c + '</td>'; }).join('') + '</tr>';
      }).join('');
    }
    set(rows);
    chartEl.parentNode.insertBefore(d, chartEl.nextSibling);
    return set;
  }
  function hbarPath(x0, y, len, th) {
    var r = Math.min(4, th / 2, Math.max(0, len));
    return 'M' + x0 + ',' + y + 'H' + (x0 + len - r) + 'Q' + (x0 + len) + ',' + y + ' ' + (x0 + len) + ',' + (y + r) +
      'V' + (y + th - r) + 'Q' + (x0 + len) + ',' + (y + th) + ' ' + (x0 + len - r) + ',' + (y + th) + 'H' + x0 + 'Z';
  }
  function niceTicks(lo, hi, n) {
    var step = Math.pow(10, Math.floor(Math.log10((hi - lo) / n)));
    [1, 2, 2.5, 5, 10].some(function (m) { if ((hi - lo) / (step * m) <= n) { step *= m; return true; } return false; });
    var out = [];
    for (var v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  function logTicks(lo, hi, sparse) {
    var out = [];
    for (var e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) {
      (sparse ? [1] : [1, 3]).forEach(function (m) { var v = m * Math.pow(10, e); if (v >= lo * 0.999 && v <= hi * 1.001) out.push(v); });
    }
    return out;
  }

  // 横向条形 / 棒棒糖图。rows: [{name, v, err, color, tip}]；o: {min, max, log, fmt, aria, lolli}
  function hbars(el, rows, o, once) {
    o = o || {};
    (once ? function (fn) { fn(); } : responsive)(function () {
      var host = plotHost(el);
      var W = Math.max(300, host.clientWidth);
      var narrow = W < 520;
      var labW = narrow ? 128 : 188, rowH = 26, m = { l: labW, r: 64, t: 6, b: 26 };
      var H = m.t + m.b + rows.length * rowH;
      var svg = s('svg', { viewBox: '0 0 ' + W + ' ' + H, width: W, height: H, role: 'img', 'aria-label': o.aria || '' }, host);
      var lo = o.min != null ? o.min : 0;
      var hi = o.max != null ? o.max : Math.max.apply(null, rows.map(function (r) { return r.v + (r.err || 0); })) * 1.05;
      var x = o.log ? logScale(lo, hi, m.l, W - m.r) : lin(lo, hi, m.l, W - m.r);
      var ticks = o.log ? logTicks(lo, hi, narrow || Math.log10(hi / lo) > 4) : niceTicks(lo, hi, narrow ? 4 : 6);
      ticks.forEach(function (t) {
        s('line', { x1: x(t), x2: x(t), y1: m.t, y2: H - m.b, 'class': t === lo && !o.log ? 'axl' : 'gl' }, svg);
        txt(svg, x(t), H - m.b + 16, o.tick ? o.tick(t) : String(t), { 'text-anchor': 'middle', 'class': 'ax' });
      });
      rows.forEach(function (r, i) {
        var y = m.t + i * rowH + 5, th = rowH - 10, cy = y + th / 2;
        var name = clip(r.name, labW - 12);
        txt(svg, labW - 8, cy + 4, name, { 'text-anchor': 'end', 'class': 'lbl' });
        var color = r.color || 'var(--text)';
        if (r.err && o.dots) {
          var a = x(Math.max(lo, r.v - r.err)), b = x(r.v + r.err);
          s('line', { x1: a, x2: b, y1: cy, y2: cy, style: 'stroke:var(--ink);stroke-width:1.5' }, svg);
          s('line', { x1: a, x2: a, y1: cy - 5, y2: cy + 5, style: 'stroke:var(--ink);stroke-width:1.5' }, svg);
          s('line', { x1: b, x2: b, y1: cy - 5, y2: cy + 5, style: 'stroke:var(--ink);stroke-width:1.5' }, svg);
        }
        if (o.dots) {
          // 轴不从 0 开始（或是对数轴）时画点，不画柱子：柱长会夸大差距
          s('line', { x1: m.l, x2: W - m.r, y1: cy, y2: cy, 'class': 'gl', style: 'stroke-dasharray:2 3' }, svg);
          s('circle', { cx: x(r.v), cy: cy, r: 5.5, style: 'fill:' + color + ';stroke:var(--surface);stroke-width:2' }, svg);
        } else {
          s('path', { d: hbarPath(x(lo), y, Math.max(1, x(r.v) - x(lo)), th), style: 'fill:' + color }, svg);
        }
        if (r.err && !o.dots) {
          var a = x(Math.max(lo, r.v - r.err)), b = x(r.v + r.err);
          s('line', { x1: a, x2: b, y1: cy, y2: cy, style: 'stroke:var(--ink);stroke-width:1.5' }, svg);
          s('line', { x1: a, x2: a, y1: cy - 5, y2: cy + 5, style: 'stroke:var(--ink);stroke-width:1.5' }, svg);
          s('line', { x1: b, x2: b, y1: cy - 5, y2: cy + 5, style: 'stroke:var(--ink);stroke-width:1.5' }, svg);
        }
        txt(svg, x(r.v + (r.err || 0)) + 8, cy + 4, o.fmt ? o.fmt(r.v) : fmt(r.v), { 'class': 'lbl' });
        var hit = s('rect', { x: 0, y: m.t + i * rowH, width: W, height: rowH, 'class': 'hit' }, svg);
        var show = function (ev) { showTip(r.tip || ('<b>' + esc(r.name) + '</b><br>' + (o.fmt ? o.fmt(r.v) : fmt(r.v))), ev); };
        hit.addEventListener('pointermove', show);
        hit.addEventListener('pointerdown', show);
        hit.addEventListener('pointerleave', hideTip);
      });
    });
  }

  // 折线图。series: [{name, color, dash, pts: [[x, y], ...]}]；o: {xlog, ylog, xmin, xmax, ymin, ymax, xticks, yticks, xfmt, yfmt, xlabel, diag, aria}
  function lines(el, series, o) {
    responsive(function () {
      var host = plotHost(el);
      var W = Math.max(300, host.clientWidth), H = o.h || 260, m = { l: 52, r: W < 520 ? 70 : 110, t: 14, b: 40 };
      var svg = s('svg', { viewBox: '0 0 ' + W + ' ' + H, width: W, height: H, role: 'img', 'aria-label': o.aria || '' }, host);
      var x = (o.xlog ? logScale : lin)(o.xmin, o.xmax, m.l, W - m.r);
      var y = (o.ylog ? logScale : lin)(o.ymin, o.ymax, H - m.b, m.t);
      (o.yticks || niceTicks(o.ymin, o.ymax, 5)).forEach(function (t) {
        s('line', { x1: m.l, x2: W - m.r, y1: y(t), y2: y(t), 'class': 'gl' }, svg);
        txt(svg, m.l - 8, y(t) + 4, o.yfmt ? o.yfmt(t) : String(t), { 'text-anchor': 'end', 'class': 'ax' });
      });
      s('line', { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, 'class': 'axl' }, svg);
      (o.xticks || niceTicks(o.xmin, o.xmax, 6)).forEach(function (t) {
        txt(svg, x(t), H - m.b + 17, o.xfmt ? o.xfmt(t) : String(t), { 'text-anchor': 'middle', 'class': 'ax' });
      });
      if (o.xlabel) txt(svg, W - m.r, H - 4, o.xlabel, { 'text-anchor': 'end', 'class': 'ax' });
      if (o.diag) {
        s('line', { x1: x(o.xmin), y1: y(o.ymin), x2: x(o.xmax), y2: y(o.ymax), style: 'stroke:var(--muted);stroke-width:1.5;stroke-dasharray:4 4' }, svg);
      }
      var dots = [], ends = [];
      series.forEach(function (se) {
        var d = se.pts.map(function (p, i) { return (i ? 'L' : 'M') + x(p[0]) + ',' + y(p[1]); }).join('');
        s('path', { d: d, style: 'fill:none;stroke:' + se.color + ';stroke-width:2' + (se.dash ? ';stroke-dasharray:5 4' : '') }, svg);
        dots.push(se.pts.map(function (p) {
          return s('circle', { cx: x(p[0]), cy: y(p[1]), r: 4, style: 'fill:' + se.color + ';stroke:var(--surface);stroke-width:2' }, svg);
        }));
        var last = se.pts[se.pts.length - 1];
        ends.push({ x: x(last[0]) + 9, y: y(last[1]) + 4, name: se.name });
      });
      // 末端标签互相挤开，至少隔 14px
      ends.sort(function (a, b) { return a.y - b.y; });
      for (var k = 1; k < ends.length; k++) if (ends[k].y - ends[k - 1].y < 14) ends[k].y = ends[k - 1].y + 14;
      ends.forEach(function (e) { txt(svg, e.x, e.y, e.name, { 'class': 'lbl' }); });
      // 十字准星 + 提示
      var xs = [];
      series.forEach(function (se) { se.pts.forEach(function (p) { if (xs.indexOf(p[0]) < 0) xs.push(p[0]); }); });
      xs.sort(function (a, b) { return a - b; });
      var cross = s('line', { y1: m.t, y2: H - m.b, style: 'stroke:var(--axis);stroke-width:1;display:none' }, svg);
      var ov = s('rect', { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, 'class': 'hit' }, svg);
      function move(ev) {
        var r = svg.getBoundingClientRect(), px = (ev.clientX - r.left) * W / r.width;
        var best = xs[0];
        xs.forEach(function (v) { if (Math.abs(x(v) - px) < Math.abs(x(best) - px)) best = v; });
        cross.setAttribute('x1', x(best)); cross.setAttribute('x2', x(best)); cross.style.display = '';
        var rows = series.map(function (se) {
          var p = se.pts.filter(function (q) { return q[0] === best; })[0];
          return p ? '<span class="k" style="color:' + se.color + '">●</span> ' + esc(se.name) + '：' + (o.yfmtTip || o.yfmt || fmt)(p[1]) : '';
        }).filter(Boolean).join('<br>');
        showTip('<b>' + (o.xname || '') + (o.xfmtTip || o.xfmt || String)(best) + '</b><br>' + rows, ev);
      }
      ov.addEventListener('pointermove', move);
      ov.addEventListener('pointerdown', move);
      ov.addEventListener('pointerleave', function () { cross.style.display = 'none'; hideTip(); });
    });
  }

  // 热力表：M 是数字矩阵，按 [lo, hi] 线性映射到颜色深浅
  function heat(el, M, rowLab, colLab, o) {
    o = o || {};
    var flat = [].concat.apply([], M);
    var lo = o.lo != null ? o.lo : Math.min.apply(null, flat), hi = o.hi != null ? o.hi : Math.max.apply(null, flat);
    var color = o.color || 'var(--text)';
    var html = '<div class="table-wrap"><table class="heat"><thead><tr><th>' + (o.corner || '') + '</th>' +
      colLab.map(function (c) { return '<th class="n">' + c + '</th>'; }).join('') + '</tr></thead><tbody>';
    M.forEach(function (row, i) {
      html += '<tr><th>' + rowLab[i] + '</th>' + row.map(function (v, j) {
        var t = (v - lo) / (hi - lo || 1);
        var a = Math.round(8 + 80 * Math.max(0, Math.min(1, t)));
        var hot = a > 55;
        return '<td class="' + (hot ? 'hot' : '') + '" style="background:color-mix(in srgb, ' + color + ' ' + a + '%, var(--surface))" title="' +
          esc(rowLab[i] + ' → ' + colLab[j]) + '">' + (o.fmt ? o.fmt(v) : v) + '</td>';
      }).join('') + '</tr>';
    });
    el.innerHTML = html + '</tbody></table></div>';
  }

  // ---------------------------------------------------------------- 表单视图（第 1、3 章）
  function docGeom() {
    var doc = D.doc, ar = doc.size[1] / doc.size[0];
    var ws = doc.words.map(function (w, i) {
      var b = w.b;
      return { i: i, t: w.t, l: w.l, e: w.e, x: b[0], y: b[1] * ar, w: Math.max(4, b[2] - b[0]), h: Math.max(4, (b[3] - b[1]) * ar),
        cx: (b[0] + b[2]) / 2, cy: (b[1] + b[3]) / 2 * ar };
    });
    var ys = ws.map(function (w) { return w.y; }), ye = ws.map(function (w) { return w.y + w.h; });
    var top = Math.max(0, Math.min.apply(null, ys) - 20), bot = Math.min(1000 * ar, Math.max.apply(null, ye) + 20);
    return { ws: ws, ar: ar, top: top, bot: bot };
  }
  function drawDoc(host, g, onPick) {
    var svg = s('svg', { viewBox: '0 ' + g.top + ' 1000 ' + (g.bot - g.top), role: 'img', 'aria-label': 'FUNSD 测试集表单：每个词一个框，颜色是标签' }, host);
    var rects = g.ws.map(function (w) {
      var r = s('rect', { x: w.x, y: w.y, width: w.w, height: w.h, rx: 2, 'class': 'w',
        style: 'fill:' + LAB_COLOR[w.l] + ';fill-opacity:0.28;stroke:' + LAB_COLOR[w.l] }, svg);
      var fs = Math.max(7, Math.min(13, w.h * 0.72));
      txt(svg, w.x + 2, w.y + w.h * 0.78, w.t, { 'class': 'wt', 'font-size': fs });
      r.addEventListener('pointermove', function (ev) {
        showTip('<b>' + esc(w.t) + '</b><br><span class="k">标签</span> ' + LAB[w.l] + '　<span class="k">阅读顺序</span> 第 ' + w.i +
          ' 个　<span class="k">实体</span> #' + w.e, ev);
      });
      r.addEventListener('pointerleave', hideTip);
      if (onPick) r.addEventListener('click', function () { onPick(w.i); });
      return r;
    });
    return { svg: svg, rects: rects };
  }

  function widgetDoc() {
    var el = document.getElementById('w-doc');
    if (!el || !D) return;
    var g = docGeom(), mode = 'label';
    var btns = el.querySelectorAll('.seg-btns button');
    var note = el.querySelector('.doc-note');
    function render() {
      var host = el.querySelector('.doc-view');
      host.innerHTML = '';
      var v = drawDoc(host, g);
      if (mode === 'order') {
        var d = g.ws.map(function (w, i) { return (i ? 'L' : 'M') + w.cx + ',' + w.cy; }).join('');
        s('path', { d: d, 'class': 'path' }, v.svg);
        g.ws.forEach(function (w) {
          if (w.i % 10 === 0) txt(v.svg, w.cx, w.y - 3, String(w.i), { 'text-anchor': 'middle', 'class': 'wt', 'font-size': 12, style: 'fill:var(--ink);font-weight:600' });
        });
        note.textContent = '细线按阅读顺序把词连起来（每 10 个词标一次序号）。同一行里左右两栏的词被串在一起，这就是一维 head 看到的“相邻”。';
      } else {
        var cnt = [0, 0, 0, 0];
        g.ws.forEach(function (w) { cnt[w.l]++; });
        note.textContent = '表单 ' + D.doc.id + '：' + g.ws.length + ' 个词，question ' + cnt[1] + '、answer ' + cnt[2] + '、header ' + cnt[3] + '、other ' + cnt[0] + '。悬停看词、标签和阅读顺序。';
      }
    }
    for (var i = 0; i < btns.length; i++) {
      btns[i].addEventListener('click', function () {
        mode = this.getAttribute('data-mode');
        for (var j = 0; j < btns.length; j++) btns[j].classList.toggle('on', btns[j] === this);
        render();
      });
    }
    render();
  }

  function widgetNeighbors() {
    var el = document.getElementById('w-neighbors');
    if (!el || !D) return;
    var g = docGeom(), sel = null, k = 3;
    var range = el.querySelector('input[type=range]'), kout = el.querySelector('.k-out'), note = el.querySelector('.doc-note');
    // 默认选一个有配对答案的 question 词
    var ansOf = {};
    D.doc.links.forEach(function (p) { ansOf[p[0]] = (ansOf[p[0]] || []).concat([p[1]]); });
    for (var i = 0; i < g.ws.length; i++) { if (g.ws[i].l === 1 && ansOf[g.ws[i].e]) { sel = i; break; } }
    function nn6(i) {
      var a = g.ws[i];
      return g.ws.filter(function (w) { return w.i !== i; })
        .sort(function (p, q) { return Math.hypot(p.cx - a.cx, p.cy - a.cy) - Math.hypot(q.cx - a.cx, q.cy - a.cy); })
        .slice(0, 6).map(function (w) { return w.i; });
    }
    function render() {
      kout.textContent = '±' + k + '（感受野 ' + (2 * k + 1) + ' 个词）';
      var host = el.querySelector('.doc-view');
      host.innerHTML = '';
      var v = drawDoc(host, g, function (i) { sel = i; render(); });
      if (sel == null) return;
      var a = g.ws[sel], near = nn6(sel), inWin = 0;
      var ans = ansOf[a.e] || [];
      g.ws.forEach(function (w) {
        var d = Math.abs(w.i - sel), r1 = d > 0 && d <= k, r2 = near.indexOf(w.i) >= 0, r3 = ans.indexOf(w.e) >= 0;
        if (r1) s('rect', { x: w.x - 3, y: w.y - 3, width: w.w + 6, height: w.h + 6, rx: 3, 'class': 'ring1' }, v.svg);
        if (r2) s('rect', { x: w.x - 6, y: w.y - 6, width: w.w + 12, height: w.h + 12, rx: 4, 'class': 'ring2' }, v.svg);
        if (r3) s('rect', { x: w.x - 1, y: w.y + w.h + 2, width: w.w + 2, height: 3, 'class': 'ring3', style: 'fill:var(--third);stroke:none' }, v.svg);
        if (r2 && d <= k) inWin++;
      });
      v.rects[sel].classList.add('sel');
      var ansWords = g.ws.filter(function (w) { return ans.indexOf(w.e) >= 0; });
      var ansDist = ansWords.length ? Math.min.apply(null, ansWords.map(function (w) { return Math.abs(w.i - sel); })) : null;
      note.innerHTML = '选中 <b>' + esc(a.t) + '</b>（' + LAB[a.l] + '，阅读顺序第 ' + a.i + ' 个）。页面上最近的 6 个词里，有 <b>' + inWin +
        '</b> 个落在阅读顺序 ±' + k + ' 以内。' + (ansDist != null ? '它配对的 answer 离它 <b>' + ansDist + '</b> 个位置' +
        (ansDist <= k ? '，在卷积感受野里。' : '，卷积 head 看不到。') : '这个词没有标注配对的 answer。');
    }
    range.addEventListener('input', function () { k = +range.value; render(); });
    render();
  }

  // ---------------------------------------------------------------- 第 6 章：阅读顺序条（派生片段 vs 金标实体）
  function widgetRuns() {
    var el = document.getElementById('w-runs');
    if (!el || !D) return;
    var ws = D.doc.words, showSep = true;
    var btns = el.querySelectorAll('.seg-btns button'), note = el.querySelector('.doc-note');
    var runs = [], i = 0;
    while (i < ws.length) {
      var j = i;
      while (j + 1 < ws.length && ws[j + 1].l === ws[i].l) j++;
      runs.push([i, j]);
      i = j + 1;
    }
    function render() {
      var host = el.querySelector('.run-strip');
      var merged = 0, field = 0;
      host.innerHTML = runs.map(function (r) {
        var l = ws[r[0]].l, ents = {};
        for (var t = r[0]; t <= r[1]; t++) ents[ws[t].e] = 1;
        var nE = Object.keys(ents).length, isM = l !== 0 && nE > 1;
        if (l !== 0) { field++; if (isM) merged++; }
        var inner = '';
        for (var u = r[0]; u <= r[1]; u++) {
          if (showSep && u > r[0] && ws[u].e !== ws[u - 1].e && l !== 0) inner += '<span class="sep" aria-hidden="true"></span>';
          inner += '<span class="lab ' + LAB_CLS[l] + '" data-i="' + u + '">' + esc(ws[u].t) + '</span>';
        }
        return '<span class="run' + (l === 0 ? ' other' : '') + (isM ? ' merged' : '') + '"' +
          (isM ? ' title="这个片段里有 ' + nE + ' 个金标实体"' : '') + '>' + inner + '</span>';
      }).join('');
      note.innerHTML = '这份表单按扁平标签派生出 <b>' + field + '</b> 个字段片段，其中 <b>' + merged +
        '</b> 个（红色虚线框）里混了不止一个金标实体' + (showSep ? '；红色竖线是金标实体的边界，扁平 tag 里没有这条信息。' : '。');
      var labs = host.querySelectorAll('.lab');
      for (var q = 0; q < labs.length; q++) {
        labs[q].addEventListener('pointermove', function (ev) {
          var w = ws[+this.getAttribute('data-i')];
          showTip('<b>' + esc(w.t) + '</b><br><span class="k">标签</span> ' + LAB[w.l] + '　<span class="k">实体</span> #' + w.e, ev);
        });
        labs[q].addEventListener('pointerleave', hideTip);
      }
    }
    for (var b = 0; b < btns.length; b++) {
      btns[b].addEventListener('click', function () {
        showSep = this.getAttribute('data-mode') === 'ent';
        for (var j = 0; j < btns.length; j++) btns[j].classList.toggle('on', btns[j] === this);
        render();
      });
    }
    render();
  }

  // ---------------------------------------------------------------- 各章的图
  function R(k) { return D.results[k]; }
  var BASE = 'linear_both';
  function rowFor(k, name) {
    var r = R(k);
    return { name: name || r.name, v: r.macro[0], err: r.macro[1], color: k === BASE ? 'var(--layout)' : 'var(--text)',
      tip: '<b>' + esc(name || r.name) + '</b><br><span class="k">宏 F1</span> ' + fmt(r.macro[0]) + ' ± ' + fmt(r.macro[1]) +
        '<br><span class="k">question / answer / header</span> ' + fmt(r.pc.question[0]) + ' / ' + fmt(r.pc.answer[0]) + ' / ' + fmt(r.pc.header[0]) +
        '<br><span class="k">ECE</span> ' + fmt(r.ece) };
  }

  function chartViews() {
    var el = document.getElementById('chart-views');
    if (!el || !D) return;
    var rows = [
      { k: 'text', name: 'text 768（HF）', v: R('linear_text').macro, kb: 768 },
      { k: 'layout', name: 'layout 192', v: D.views.linear_layout.macro, kb: 192 },
      { k: 'both', name: 'both 960（官方）', v: R('linear_both').macro, kb: 960 },
      { k: 'last4', name: 'last4 3264', v: D.views.linear_last4.macro, kb: 3264 }
    ].map(function (r) {
      return { name: r.name, v: r.v[0], err: r.v[1], color: r.k === 'both' ? 'var(--layout)' : 'var(--text)',
        tip: '<b>' + r.name + '</b><br><span class="k">宏 F1</span> ' + fmt(r.v[0]) + ' ± ' + fmt(r.v[1]) +
          '<br><span class="k">中位文档 142 词，fp16 传输</span> ' + bytes(142 * r.kb * 2) };
    });
    hbars(el, rows, { min: 0.5, max: 0.72, aria: '四种输出契约下线性 head 的宏 F1', tick: function (t) { return t.toFixed(2); }, dots: true });
  }

  function chartRel(id, keys) {
    var el = document.getElementById(id);
    if (!el || !D) return;
    var colors = ['var(--text)', 'var(--layout)'];
    var names = Object.keys(D.reliability).filter(function (n) { return !keys || keys.indexOf(n) >= 0; });
    var series = names.map(function (n, i) {
      return { name: n + ' · ECE ' + fmt(D.reliability[n].ece), color: colors[i],
        pts: D.reliability[n].rows.map(function (r) { return [r.conf, r.acc]; }) };
    });
    legend(el, series.map(function (se) { return { name: se.name, color: se.color }; }).concat([{ name: '完美校准（对角线）', color: 'var(--muted)' }]));
    series.forEach(function (se) { se.name = se.name.split(' · ')[0]; });
    lines(el, series, { xmin: 0.3, xmax: 1, ymin: 0.3, ymax: 1, diag: true, xname: '平均置信度 ', aria: '置信度可靠性图',
      xfmt: function (t) { return t.toFixed(1); }, yfmt: function (t) { return t.toFixed(1); }, yfmtTip: function (t) { return '准确率 ' + fmt(t); },
      xfmtTip: function (t) { return t.toFixed(3); }, xlabel: '平均置信度 →', h: 280 });
    var rows = [];
    names.forEach(function (n) {
      D.reliability[n].rows.forEach(function (r) {
        rows.push([esc(n), '(' + r.lo.toFixed(2) + ', ' + r.hi.toFixed(2) + ']', r.n, fmt(r.conf), fmt(r.acc)]);
      });
    });
    dataTable(el, '查看数据表', ['置信度来源', '区间', '词数', '平均置信度', '实际准确率'], rows);
  }

  function heatCM() {
    var el = document.getElementById('cm-linear');
    if (!el || !D) return;
    var M = D.linear_cm.map(function (row) {
      var tot = row.reduce(function (a, b) { return a + b; }, 0);
      return row.map(function (v) { return v / tot; });
    });
    heat(el, M, LAB.map(function (l) { return '真值 ' + l; }), LAB.map(function (l) { return '预测 ' + l; }),
      { lo: 0, hi: 1, fmt: function (v) { return (100 * v).toFixed(0) + '%'; }, corner: '按行归一' });
  }

  function heatTrans() {
    var el = document.getElementById('trans-crf');
    if (!el || !D) return;
    heat(el, D.crf_trans, LAB.map(function (l) { return '前 ' + l; }), LAB.map(function (l) { return '后 ' + l; }),
      { fmt: function (v) { return v.toFixed(2); }, color: 'var(--layout)', corner: 'trans[前, 后]' });
  }

  function chartSeq() {
    var el = document.getElementById('chart-seq');
    if (!el || !D) return;
    hbars(el, [rowFor('linear_both', 'linear（基线）'), rowFor('conv1d'), rowFor('bilstm'), rowFor('transformer')],
      { min: 0.6, max: 0.78, aria: '序列 head 与线性基线的宏 F1', tick: function (t) { return t.toFixed(2); }, dots: true });
  }

  function chartLat(id, keys, names) {
    var el = document.getElementById(id);
    if (!el || !D) return;
    var colors = ['var(--text)', 'var(--layout)', 'var(--third)'];
    var ci = 0;
    var series = keys.map(function (k, i) {
      var isBase = k === 'linear_both';
      return { name: names ? names[i] : R(k).name, color: isBase ? 'var(--muted)' : colors[ci++], dash: isBase,
        pts: [64, 128, 256, 512].map(function (n) { return [n, R(k).lat[n]]; }) };
    });
    var all = [].concat.apply([], series.map(function (se) { return se.pts.map(function (p) { return p[1]; }); }));
    var lo = Math.pow(10, Math.floor(Math.log10(Math.min.apply(null, all))));
    var hi = Math.pow(10, Math.ceil(Math.log10(Math.max.apply(null, all))));
    legend(el, series.map(function (se) { return { name: se.name, color: se.color }; }));
    lines(el, series, { xlog: true, ylog: true, xmin: 64, xmax: 512, ymin: lo, ymax: hi, xticks: [64, 128, 256, 512],
      yticks: logTicks(lo, hi), yfmt: function (t) { return t >= 1 ? String(t) : String(+t.toPrecision(1)); },
      yfmtTip: ms, xname: 'n = ', xlabel: '一份文档的词数 n →', aria: '单线程 head 延迟随词数的变化' });
  }

  function chartLinkDist() {
    var el = document.getElementById('chart-linkdist');
    if (!el || !D) return;
    var d = D.neighbors.link_dist, half = Math.floor(D.neighbors.rf / 2);
    var bins = [['紧挨着（=1）', 1, 1], ['2 ~ ' + half, 2, half], [(half + 1) + ' ~ 10', half + 1, 10], ['> 10', 11, 1e9]];
    var rows = bins.map(function (b, i) {
      var n = d.filter(function (v) { return v >= b[1] && v <= b[2]; }).length;
      return { name: b[0], v: n / d.length, color: i < 2 ? 'var(--text)' : 'var(--layout)',
        tip: '<b>' + b[0] + '</b><br>' + n + ' 对（' + (100 * n / d.length).toFixed(1) + '%）' };
    });
    hbars(el, rows, { min: 0, max: 0.7, aria: 'question 与 answer 在阅读顺序中的距离分布', fmt: function (v) { return (100 * v).toFixed(1) + '%'; },
      tick: function (t) { return Math.round(100 * t) + '%'; } });
  }

  function chartLayout() {
    var el = document.getElementById('chart-layout');
    if (!el || !D) return;
    hbars(el, [rowFor('linear_both', 'linear（基线）'), rowFor('transformer', 'Transformer（无 2D 偏置）'), rowFor('layout2d', 'Transformer + 2D 偏置')],
      { min: 0.6, max: 0.78, aria: '二维相对位置偏置的消融', tick: function (t) { return t.toFixed(2); }, dots: true });
  }

  function chartBias() {
    var el = document.getElementById('chart-bias');
    if (!el || !D) return;
    var side = D.bias.side;
    function dist(b) { var a = Math.abs(b); return a === 0 ? 0 : Math.round(2 * (Math.pow(2, a / 2) - 1)) * (b < 0 ? -1 : 1); }
    var series = [['横向 dx', D.bias.dx, 'var(--text)'], ['纵向 dy', D.bias.dy, 'var(--layout)']].map(function (p) {
      return { name: p[0], color: p[2], pts: p[1].map(function (v, i) { return [i - (side - 1), v]; }) };
    });
    var all = D.bias.dx.concat(D.bias.dy);
    var lo = Math.floor(Math.min.apply(null, all) * 4) / 4, hi = Math.ceil(Math.max.apply(null, all) * 4) / 4;
    legend(el, series.map(function (se) { return { name: se.name, color: se.color }; }));
    lines(el, series, { xmin: -(side - 1), xmax: side - 1, ymin: lo, ymax: hi, xticks: [-15, -10, -6, -2, 0, 2, 6, 10, 15],
      xfmt: function (b) { return String(dist(b)); }, xfmtTip: function (b) { return '桶 ' + b + '（距离约 ' + dist(b) + ' 起）'; },
      yfmt: function (t) { return t.toFixed(2); }, xlabel: '词 j 相对词 i 的距离（0..1000 单位）→', aria: '学到的二维相对位置偏置' });
  }

  function chartSpanEval() {
    var el = document.getElementById('chart-span');
    if (!el || !D) return;
    var names = Object.keys(D.span_eval);
    var rows = [];
    names.forEach(function (n) {
      rows.push({ name: n.replace('（上限）', '') + ' · 对片段', v: D.span_eval[n].vs_runs[2], color: 'var(--text)' });
      rows.push({ name: n.replace('（上限）', '') + ' · 对实体', v: D.span_eval[n].vs_entities[2], color: 'var(--layout)' });
    });
    legend(el, [{ name: '对派生片段的 F1', color: 'var(--text)', sq: true }, { name: '对金标实体的 F1', color: 'var(--layout)', sq: true }]);
    hbars(el, rows, { min: 0, max: 1, aria: '片段级 exact match F1', tick: function (t) { return t.toFixed(1); } });
  }

  function chartSpanCmp() {
    var el = document.getElementById('chart-spancmp');
    if (!el || !D) return;
    hbars(el, [rowFor('linear_both', 'linear（基线）'), rowFor('crf', 'linear + CRF'), rowFor('span', 'biaffine 片段')],
      { min: 0.6, max: 0.78, aria: '片段 head 与基线的逐词宏 F1', tick: function (t) { return t.toFixed(2); }, dots: true });
  }

  function chartCrf() {
    var el = document.getElementById('chart-crf');
    if (!el || !D) return;
    hbars(el, [rowFor('linear_both', 'linear（无 CRF）'), rowFor('crf', 'linear + CRF')],
      { min: 0.6, max: 0.78, aria: 'CRF 消融的宏 F1', tick: function (t) { return t.toFixed(2); }, dots: true });
  }

  function chartTree() {
    var el = document.getElementById('chart-tree');
    if (!el || !D) return;
    hbars(el, [rowFor('linear_both', 'linear（基线）'), rowFor('tree'), rowFor('tree_pca')],
      { min: 0.6, max: 0.78, aria: '树模型 head 的宏 F1', tick: function (t) { return t.toFixed(2); }, dots: true });
  }

  function chartProto() {
    var el = document.getElementById('chart-proto');
    if (!el || !D) return;
    hbars(el, [rowFor('linear_both', 'linear（基线）'), rowFor('ncm'), rowFor('knn'), rowFor('protoner')],
      { min: 0.4, max: 0.78, aria: '原型与相似度 head 的宏 F1', tick: function (t) { return t.toFixed(2); }, dots: true });
  }

  function chartFewshot() {
    var el = document.getElementById('chart-fewshot');
    if (!el || !D) return;
    var c = D.fewshot.curve;
    var series = [['linear（重新训练）', c.linear, 'var(--text)'], ['NCM 类均值', c.NCM, 'var(--layout)'], ['kNN k=10', c.kNN, 'var(--third)']]
      .map(function (p) { return { name: p[0], color: p[2], pts: c.sizes.map(function (n, i) { return [n, p[1][i]]; }) }; });
    legend(el, series.map(function (se) { return { name: se.name, color: se.color }; }));
    var ymin = Math.floor(Math.min.apply(null, [].concat(c.linear, c.NCM, c.kNN)) * 10) / 10;
    lines(el, series, { xlog: true, xmin: 1, xmax: c.sizes[c.sizes.length - 1], ymin: ymin, ymax: 0.75, xticks: c.sizes,
      yfmt: function (t) { return t.toFixed(2); }, xname: '训练文档 ', xfmtTip: function (n) { return n + ' 份'; },
      xlabel: '训练文档数（对数轴）→', aria: '少样本学习曲线' });
  }

  // ---------------------------------------------------------------- 第 9 章：总表与多任务成本
  var METRICS = {
    macro: { name: '宏 F1', get: function (r) { return r.macro[0]; }, err: function (r) { return r.macro[1]; }, hi: true, fmt: function (v) { return fmt(v); } },
    header: { name: 'header F1', get: function (r) { return r.pc.header[0]; }, err: function (r) { return r.pc.header[1]; }, hi: true, fmt: function (v) { return fmt(v); } },
    ent: { name: '实体全对率', get: function (r) { return r.ent; }, hi: true, fmt: function (v) { return fmt(v); } },
    ece: { name: 'ECE（越低越好）', get: function (r) { return r.ece; }, hi: false, fmt: function (v) { return fmt(v); } },
    lat: { name: '单线程延迟 n=256', get: function (r) { return r.lat[256]; }, hi: false, log: true, fmt: ms },
    size: { name: '每个任务的文件', get: function (r) { return r.bytes; }, hi: false, log: true, fmt: bytes }
  };
  function chartAll() {
    var el = document.getElementById('chart-all');
    if (!el || !D) return;
    var sel = el.querySelector('select'), keys = D.compare.rows.map(function (r) { return r[0]; });
    function render() {
      var M = METRICS[sel.value];
      var rows = keys.map(function (k) {
        var r = R(k);
        return { k: k, name: r.name, v: M.get(r), err: M.err ? M.err(r) : 0, color: k === BASE ? 'var(--layout)' : 'var(--text)',
          tip: '<b>' + esc(r.name) + '</b>（' + r.fam + '）<br><span class="k">宏 F1</span> ' + fmt(r.macro[0]) + ' ± ' + fmt(r.macro[1]) +
            '<br><span class="k">ECE</span> ' + fmt(r.ece) + '<br><span class="k">延迟 n=256</span> ' + ms(r.lat[256]) +
            '<br><span class="k">文件</span> ' + bytes(r.bytes) };
      }).sort(function (a, b) { return M.hi ? b.v - a.v : a.v - b.v; });
      var vals = rows.map(function (r) { return r.v; });
      var o = { aria: '所有 head 的' + M.name, fmt: M.fmt, dots: sel.value !== 'ece' };
      if (M.log) {
        o.log = true;
        o.min = Math.pow(10, Math.floor(Math.log10(Math.min.apply(null, vals))));
        o.max = Math.pow(10, Math.ceil(Math.log10(Math.max.apply(null, vals)) + 0.3));
        o.tick = function (t) { return M.fmt(t).replace(/\.0+ /, ' '); };
      } else if (sel.value === 'ece') {
        o.min = 0; o.max = Math.max.apply(null, vals) * 1.15; o.tick = function (t) { return t.toFixed(2); };
      } else {
        o.min = 0.3; o.max = 0.85; o.tick = function (t) { return t.toFixed(1); };
      }
      hbars(el, rows, o, true);
      if (setTable) setTable(tableRows());
    }
    function tableRows() {
      return keys.map(function (k) {
        var r = R(k);
        return [esc(r.name), fmt(r.macro[0]) + ' ± ' + fmt(r.macro[1]), fmt(r.pc.header[0]), fmt(r.ent), fmt(r.ece), ms(r.lat[256]), bytes(r.bytes)];
      });
    }
    var setTable = null;
    sel.addEventListener('change', render);
    responsive(render);
    setTable = dataTable(el, '查看数据表（14 个 head）', ['head', '宏 F1', 'header F1', '实体全对', 'ECE', '延迟 n=256', '文件'], tableRows());
  }

  function widgetCost() {
    var el = document.getElementById('w-cost');
    if (!el || !D) return;
    var range = el.querySelector('input[type=range]'), out = el.querySelector('.t-out'), stat = el.querySelector('.cost-stats');
    var keys = D.compare.rows.map(function (r) { return r[0]; });
    var bb = D.compare.backbone_params * 4;
    var chart = el.querySelector('.chart');
    function render() {
      var T = +range.value;
      out.textContent = T + ' 个任务';
      var rows = keys.map(function (k) {
        var r = R(k), tot = T * r.bytes;
        return { name: r.name, v: tot, color: k === BASE ? 'var(--layout)' : 'var(--text)',
          tip: '<b>' + esc(r.name) + '</b><br>' + T + ' × ' + bytes(r.bytes) + ' = ' + bytes(tot) + '<br><span class="k">相当于 backbone 的</span> ' + (100 * tot / bb).toFixed(1) + '%' };
      }).sort(function (a, b) { return a.v - b.v; });
      stat.innerHTML =
        '<div class="stat"><div class="s-label">共享 backbone（fp32，一份）</div><div class="s-value">' + bytes(bb) + '</div><div class="s-note">' + (D.compare.backbone_params / 1e6).toFixed(1) + 'M 参数</div></div>' +
        '<div class="stat"><div class="s-label">每个任务各存一份微调 backbone</div><div class="s-value">' + bytes(T * bb) + '</div><div class="s-note">' + T + ' × ' + bytes(bb) + '，作对照</div></div>';
      var vals = rows.map(function (r) { return r.v; });
      hbars(chart, rows, { log: true, dots: true, fmt: bytes, min: Math.pow(10, Math.floor(Math.log10(Math.min.apply(null, vals)))),
        max: Math.pow(10, Math.ceil(Math.log10(Math.max.apply(null, vals)) + 0.3)), aria: T + ' 个任务时各类 head 的总存储',
        tick: function (t) { return bytes(t).replace(/\.0+ /, ' '); } }, true);
    }
    range.addEventListener('input', render);
    responsive(render);
  }

  // ---------------------------------------------------------------- 启动
  buildNav();
  buildPager();
  highlightAll();
  if (D) {
    [widgetDoc, widgetNeighbors, widgetRuns, chartViews, heatCM, heatTrans, chartSeq, chartLinkDist, chartLayout, chartBias,
      chartSpanEval, chartSpanCmp, chartCrf, chartTree, chartProto, chartFewshot, chartAll, widgetCost].forEach(function (fn) {
      try { fn(); } catch (e) { if (window.console) console.error(fn.name, e); }
    });
    chartRel('chart-rel', ['softmax（线性）']);
    chartRel('chart-rel2');
    chartLat('chart-lat', ['linear_both', 'conv1d', 'bilstm', 'transformer'], ['linear', 'Conv1D', 'BiLSTM', 'Transformer']);
    chartLat('chart-lat-crf', ['linear_both', 'crf'], ['linear', 'linear + CRF']);
    chartLat('chart-lat-layout', ['linear_both', 'transformer', 'layout2d'], ['linear', 'Transformer', '+ 2D 偏置']);
    chartLat('chart-lat-span', ['linear_both', 'span'], ['linear', 'biaffine 片段']);
    chartLat('chart-lat-tree', ['linear_both', 'tree', 'tree_pca'], ['linear', 'GBDT 964', 'GBDT PCA64']);
    chartLat('chart-lat-proto', ['linear_both', 'ncm', 'knn'], ['linear', 'NCM', 'kNN']);
  }
})();
