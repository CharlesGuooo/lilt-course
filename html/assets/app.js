/* LiLT 架构课 · 共用脚本
   目录与翻页、主题切换、Python 代码高亮、提示框、数据图、交互组件。
   数据来自 data.js（window.LILT），由真实权重跑出来；没有 data.js 的页面不会初始化依赖它的组件。 */
(function () {
  'use strict';

  var D = window.LILT || null;

  var CHAPTERS = [
    { f: 'index.html', n: '··', title: '课程首页' },
    { f: 'ch00_overview.html', n: '00', title: '全局鸟瞰' },
    { f: 'ch01_roberta.html', n: '01', title: 'RoBERTa 复习' },
    { f: 'ch02_inputs.html', n: '02', title: '输入准备' },
    { f: 'ch03_layout_embedding.html', n: '03', title: 'Layout embedding' },
    { f: 'ch04_biacm.html', n: '04', title: '双流 encoder 与 BiACM' },
    { f: 'ch05_pretraining.html', n: '05', title: '预训练' },
    { f: 'ch06_heads.html', n: '06', title: '输出与任务头' },
    { f: 'ch07_frozen_backbone.html', n: '07', title: '落地：冻结 backbone' }
  ];
  var page = document.body.getAttribute('data-page') || 'index.html';
  var cur = -1;
  CHAPTERS.forEach(function (c, i) { if (c.f === page) cur = i; });

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
  function fmt(v, d) { return Number(v).toFixed(d == null ? 3 : d); }
  function tokLabel(i) {
    var t = D.tokens[i];
    return t.wid == null ? t.t : t.t + ' · ' + D.words[t.wid].w;
  }

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
      '<span><b>LiLT 架构课</b><small>双流 Transformer 拆解</small></span></a>' +
      '<nav aria-label="章节目录"><ul class="toc">' + items + '</ul></nav>' +
      '<div class="side-foot"><span class="legend-inline"><i style="background:var(--text)"></i>文本流</span>' +
      '<span class="legend-inline"><i style="background:var(--layout)"></i>layout 流</span><br>' +
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
    '\\b(def|return|for|in|if|elif|else|import|from|with|as|lambda|not|and|or|is|None|True|False|class|while|try|except|raise|pass|assert)\\b',
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
  // 柱子：数据端 4px 圆角，基线端方角
  function barPath(x, yTop, w, yBase) {
    var hgt = yBase - yTop, r = Math.min(4, w / 2, Math.max(0, hgt));
    return 'M' + x + ',' + yBase + 'V' + (yTop + r) + 'Q' + x + ',' + yTop + ' ' + (x + r) + ',' + yTop +
      'H' + (x + w - r) + 'Q' + (x + w) + ',' + yTop + ' ' + (x + w) + ',' + (yTop + r) + 'V' + yBase + 'Z';
  }
  function hbarPath(x0, y, len, th) {
    var r = Math.min(4, th / 2, Math.max(0, len));
    return 'M' + x0 + ',' + y + 'H' + (x0 + len - r) + 'Q' + (x0 + len) + ',' + y + ' ' + (x0 + len) + ',' + (y + r) +
      'V' + (y + th - r) + 'Q' + (x0 + len) + ',' + (y + th) + ' ' + (x0 + len - r) + ',' + (y + th) + 'H' + x0 + 'Z';
  }
  function yAxis(svg, y, ticks, x0, x1, f) {
    ticks.forEach(function (t) {
      s('line', { x1: x0, x2: x1, y1: y(t), y2: y(t), 'class': t === 0 ? 'axl' : 'gl' }, svg);
      txt(svg, x0 - 8, y(t) + 4, f ? f(t) : String(t), { 'text-anchor': 'end', 'class': 'ax' });
    });
  }

  // ---------------------------------------------------------------- 第 1 章：纯 RoBERTa vs LiLT 逐 token 余弦相似度（点图）
  function chartRobertaCos() {
    var el = document.getElementById('chart-cos');
    if (!el || !D) return;
    var ys = D.roberta_vs_lilt.per_token, n = ys.length;
    var low = ys.map(function (v, i) { return [v, i]; }).sort(function (a, b) { return a[0] - b[0]; }).slice(0, 3)
      .map(function (p) { return p[1]; });
    responsive(function () {
      var host = plotHost(el);
      var W = Math.max(300, host.clientWidth), H = 240, m = { l: 44, r: 14, t: 16, b: 34 };
      var svg = s('svg', { viewBox: '0 0 ' + W + ' ' + H, width: W, height: H, role: 'img',
        'aria-label': '纯 RoBERTa 与 LiLT 文本流最后一层逐 token 余弦相似度' }, host);
      var x = lin(0, n - 1, m.l + 6, W - m.r - 6), y = lin(0.9, 1.0, H - m.b, m.t);
      yAxis(svg, y, [0.9, 0.925, 0.95, 0.975, 1.0], m.l, W - m.r, function (t) { return t.toFixed(3); });
      [0, 10, 20, 30, 40, 50, 60, n - 1].forEach(function (t) {
        txt(svg, x(t), H - m.b + 18, String(t), { 'text-anchor': 'middle', 'class': 'ax' });
      });
      txt(svg, W - m.r, H - 4, 'token 下标 →', { 'text-anchor': 'end', 'class': 'ax' });
      var dots = ys.map(function (v, i) {
        return s('circle', { cx: x(i), cy: y(v), r: 4, style: 'fill:var(--text);stroke:var(--surface);stroke-width:2' }, svg);
      });
      low.forEach(function (i) {
        txt(svg, x(i) + 8, y(ys[i]) + 4, D.tokens[i].t, { 'class': 'lbl' });
      });
      var ov = s('rect', { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, 'class': 'hit' }, svg);
      var on = -1;
      function move(ev) {
        var r = svg.getBoundingClientRect();
        var i = Math.round(x.inv((ev.clientX - r.left) * W / r.width));
        i = Math.max(0, Math.min(n - 1, i));
        if (on >= 0) dots[on].setAttribute('r', 4);
        dots[i].setAttribute('r', 6); on = i;
        showTip('<b>' + esc(tokLabel(i)) + '</b><br><span class="k">token ' + i + ' · 余弦相似度</span> ' + fmt(ys[i]), ev);
      }
      ov.addEventListener('pointermove', move);
      ov.addEventListener('pointerdown', move);
      ov.addEventListener('pointerleave', function () { if (on >= 0) dots[on].setAttribute('r', 4); on = -1; hideTip(); });
    });
    dataTable(el, '查看数据表（66 个 token）', ['token', '所属词', '余弦相似度'], ys.map(function (v, i) {
      var t = D.tokens[i];
      return [esc(t.t), t.wid == null ? '—' : esc(D.words[t.wid].w), fmt(v)];
    }));
  }

  // ---------------------------------------------------------------- 第 3 章：坐标表平滑性（折线图）
  function chartCoord() {
    var el = document.getElementById('chart-coord');
    if (!el || !D) return;
    var c = D.coord_cos;
    var series = [
      { name: 'x 坐标表：cos(x=300, x=k)', color: 'var(--layout)', ys: c.x },
      { name: 'y 坐标表：cos(y=300, y=k)', color: 'var(--third)', ys: c.y }
    ];
    legend(el, series.map(function (sr) { return { name: sr.name, color: sr.color }; }));
    responsive(function () {
      var host = plotHost(el);
      var W = Math.max(300, host.clientWidth), H = 270, m = { l: 44, r: 14, t: 14, b: 34 };
      var svg = s('svg', { viewBox: '0 0 ' + W + ' ' + H, width: W, height: H, role: 'img',
        'aria-label': '坐标表第 300 行与其他行的余弦相似度' }, host);
      var x = lin(0, 1000, m.l, W - m.r), y = lin(-0.6, 1.0, H - m.b, m.t);
      yAxis(svg, y, [-0.5, 0, 0.5, 1.0], m.l, W - m.r, function (t) { return t.toFixed(1); });
      [0, 200, 400, 600, 800, 1000].forEach(function (t) {
        txt(svg, x(t), H - m.b + 18, String(t), { 'text-anchor': 'middle', 'class': 'ax' });
      });
      txt(svg, W - m.r, H - 4, '坐标值 k →', { 'text-anchor': 'end', 'class': 'ax' });
      s('line', { x1: x(300), x2: x(300), y1: m.t, y2: H - m.b, 'class': 'axl' }, svg);
      txt(svg, x(300) + 6, m.t + 10, '参照点 300', { 'class': 'lbl' });
      series.forEach(function (sr) {
        var d = sr.ys.map(function (v, i) { return (i ? 'L' : 'M') + x(c.ks[i]).toFixed(1) + ',' + y(v).toFixed(1); }).join('');
        s('path', { d: d, fill: 'none', style: 'stroke:' + sr.color + ';stroke-width:2;stroke-linejoin:round;stroke-linecap:round' }, svg);
      });
      var cross = s('line', { y1: m.t, y2: H - m.b, 'class': 'axl', style: 'display:none' }, svg);
      var dots = series.map(function (sr) {
        return s('circle', { r: 4.5, style: 'fill:' + sr.color + ';stroke:var(--surface);stroke-width:2;display:none' }, svg);
      });
      var ov = s('rect', { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, 'class': 'hit' }, svg);
      function move(ev) {
        var r = svg.getBoundingClientRect();
        var k = x.inv((ev.clientX - r.left) * W / r.width);
        var i = Math.max(0, Math.min(c.ks.length - 1, Math.round(k / 5)));
        var X = x(c.ks[i]);
        cross.setAttribute('x1', X); cross.setAttribute('x2', X); cross.style.display = '';
        dots.forEach(function (d, j) { d.setAttribute('cx', X); d.setAttribute('cy', y(series[j].ys[i])); d.style.display = ''; });
        showTip('<b>k = ' + c.ks[i] + '</b><br><span class="k">x 表</span> ' + fmt(c.x[i]) + '<br><span class="k">y 表</span> ' + fmt(c.y[i]), ev);
      }
      ov.addEventListener('pointermove', move);
      ov.addEventListener('pointerdown', move);
      ov.addEventListener('pointerleave', function () {
        cross.style.display = 'none';
        dots.forEach(function (d) { d.style.display = 'none'; });
        hideTip();
      });
    });
    var rows = [];
    for (var i = 0; i < c.ks.length; i += 4) rows.push([c.ks[i], fmt(c.x[i]), fmt(c.y[i])]);
    dataTable(el, '查看数据表（每 20 取一个点）', ['k', 'x 表', 'y 表'], rows);
  }

  // ---------------------------------------------------------------- 第 4 章：每层 layout 分数的相对分量（柱图）
  function chartRatio() {
    var el = document.getElementById('chart-ratio');
    if (!el || !D) return;
    var L = D.attn.layers;
    var vals = L.map(function (l) { return l.std_layout / l.std_text; });
    responsive(function () {
      var host = plotHost(el);
      var W = Math.max(300, host.clientWidth), H = 240, m = { l: 40, r: 10, t: 18, b: 34 };
      var svg = s('svg', { viewBox: '0 0 ' + W + ' ' + H, width: W, height: H, role: 'img',
        'aria-label': '每一层 layout 分数标准差与文本分数标准差之比' }, host);
      var y = lin(0, 0.8, H - m.b, m.t);
      yAxis(svg, y, [0, 0.2, 0.4, 0.6, 0.8], m.l, W - m.r, function (t) { return t.toFixed(1); });
      var band = (W - m.l - m.r) / vals.length, bw = Math.min(24, band * 0.62);
      vals.forEach(function (v, i) {
        var bx = m.l + band * i + (band - bw) / 2;
        s('path', { d: barPath(bx, y(v), bw, y(0)), style: 'fill:var(--layout)' }, svg);
        txt(svg, m.l + band * (i + 0.5), H - m.b + 18, String(i), { 'text-anchor': 'middle', 'class': 'ax' });
        if (i === 0 || i === vals.length - 1) {
          txt(svg, m.l + band * (i + 0.5), y(v) - 6, v.toFixed(2), { 'text-anchor': 'middle', 'class': 'lbl-strong' });
        }
        var hit = s('rect', { x: m.l + band * i, y: m.t, width: band, height: H - m.t - m.b, 'class': 'hit' }, svg);
        function show(ev) {
          showTip('<b>第 ' + i + ' 层</b><br><span class="k">std(文本分数)</span> ' + fmt(L[i].std_text) +
            '<br><span class="k">std(layout 分数)</span> ' + fmt(L[i].std_layout) + '<br><span class="k">比值</span> ' + v.toFixed(2), ev);
        }
        hit.addEventListener('pointermove', show);
        hit.addEventListener('pointerdown', show);
        hit.addEventListener('pointerleave', hideTip);
      });
      txt(svg, W - m.r, H - 4, '层 →', { 'text-anchor': 'end', 'class': 'ax' });
    });
    dataTable(el, '查看数据表', ['层', 'std(文本分数)', 'std(layout 分数)', '比值'], L.map(function (l, i) {
      return [i, fmt(l.std_text), fmt(l.std_layout), vals[i].toFixed(2)];
    }));
  }

  // ---------------------------------------------------------------- 第 4 章：'$1,250.00' 的注意力，逐层可选（成对横条）
  function widgetAttn() {
    var w = document.getElementById('w-attn');
    if (!w || !D) return;
    var range = w.querySelector('input[type=range]'), out = w.querySelector('.layer-out');
    var el = w.querySelector('.chart'), note = w.querySelector('.attn-note');
    var layer = Number(range.value), qi = D.attn.query_index;
    var series = [
      { name: '只用文本分数', color: 'var(--text)' },
      { name: '文本 + layout 分数（LiLT 实际）', color: 'var(--layout)' }
    ];
    legend(el, series.map(function (sr) { return { name: sr.name, color: sr.color, sq: true }; }));
    var setRows = null;
    function rows() {
      var l = D.attn.layers[layer], idx = [];
      for (var i = 0; i < l.p_text.length; i++) idx.push(i);
      idx.sort(function (a, b) { return Math.max(l.p_both[b], l.p_text[b]) - Math.max(l.p_both[a], l.p_text[a]); });
      return idx.slice(0, 8).map(function (i) {
        return { i: i, label: tokLabel(i) + (i === qi ? '（自己）' : ''), a: l.p_text[i], b: l.p_both[i] };
      });
    }
    function render() {
      var host = plotHost(el), rs = rows();
      var W = Math.max(300, host.clientWidth), rowH = 30, m = { l: Math.min(190, W * 0.42), r: 44, t: 4, b: 26 };
      var H = m.t + rs.length * rowH + m.b;
      var maxv = Math.max(0.2, Math.ceil(Math.max.apply(null, rs.map(function (r) { return Math.max(r.a, r.b); })) * 10) / 10);
      var svg = s('svg', { viewBox: '0 0 ' + W + ' ' + H, width: W, height: H, role: 'img',
        'aria-label': '第 ' + layer + ' 层查询 token 的注意力分布' }, host);
      var x = lin(0, maxv, m.l, W - m.r);
      var ticks = [];
      for (var t = 0; t <= maxv + 1e-9; t += maxv > 0.4 ? 0.2 : 0.1) ticks.push(Math.round(t * 10) / 10);
      ticks.forEach(function (t) {
        s('line', { x1: x(t), x2: x(t), y1: m.t, y2: H - m.b, 'class': t === 0 ? 'axl' : 'gl' }, svg);
        txt(svg, x(t), H - 8, t.toFixed(1), { 'text-anchor': 'middle', 'class': 'ax' });
      });
      rs.forEach(function (r, k) {
        var y0 = m.t + k * rowH + 5, th = 9;
        var maxc = Math.max(8, Math.floor((m.l - 10) / 7.4));
        var label = r.label.length > maxc ? r.label.slice(0, maxc - 1) + '…' : r.label;
        txt(svg, m.l - 8, y0 + th + 4, label, { 'text-anchor': 'end', 'class': 'lbl', style: 'font-family:var(--mono)' });
        s('path', { d: hbarPath(x(0), y0, x(r.a) - x(0), th), style: 'fill:var(--text)' }, svg);
        s('path', { d: hbarPath(x(0), y0 + th + 2, x(r.b) - x(0), th), style: 'fill:var(--layout)' }, svg);
        var hit = s('rect', { x: 0, y: m.t + k * rowH, width: W, height: rowH, 'class': 'hit' }, svg);
        function show(ev) {
          var d = r.b - r.a;
          showTip('<b>' + esc(r.label) + '</b><br><span class="k">只用文本</span> ' + fmt(r.a) +
            '<br><span class="k">文本 + layout</span> ' + fmt(r.b) + '<br><span class="k">变化</span> ' + (d >= 0 ? '+' : '') + fmt(d), ev);
        }
        hit.addEventListener('pointermove', show);
        hit.addEventListener('pointerdown', show);
        hit.addEventListener('pointerleave', hideTip);
      });
      var tr = rs.map(function (r) { return [esc(r.label), fmt(r.a), fmt(r.b), (r.b - r.a >= 0 ? '+' : '') + fmt(r.b - r.a)]; });
      if (setRows) setRows(tr); else setRows = dataTable(el, '查看数据表（当前层）', ['token · 所属词', '只用文本', '文本 + layout', '变化'], tr);
      // 按 segment 汇总：同框（查询词自己的 segment）、同一行左边的 key、同一列的其他金额
      var l = D.attn.layers[layer], selfSeg = D.words[D.tokens[qi].wid].seg;
      function segSum(segs) {
        var a = 0, b2 = 0;
        D.tokens.forEach(function (t, i) {
          if (t.wid != null && segs.indexOf(D.words[t.wid].seg) >= 0) { a += l.p_text[i]; b2 += l.p_both[i]; }
        });
        return '<code>' + fmt(a) + ' → ' + fmt(b2) + '</code>';
      }
      var sameRow = [], sameCol = [];
      D.segments.forEach(function (sg, k) {
        if (k === selfSeg) return;
        var me = D.segments[selfSeg].box;
        if (Math.abs(sg.box[1] - me[1]) < 10) sameRow.push(k);
        else if (sg.box[0] > 600 && sg.text.charAt(0) === '$') sameCol.push(k);
      });
      note.innerHTML = '第 ' + layer + ' 层，按 segment 汇总的注意力（只用文本 → 文本 + layout）：同框的 <b>' +
        esc(D.segments[selfSeg].text) + '</b> ' + segSum([selfSeg]) + '；同一行左边的 <b>' +
        sameRow.map(function (k) { return esc(D.segments[k].text); }).join('、') + '</b> ' + segSum(sameRow) +
        '；同一列的其他金额 <b>' + sameCol.map(function (k) { return esc(D.segments[k].text); }).join('、') + '</b> ' + segSum(sameCol) + '。';
    }
    range.addEventListener('input', function () {
      layer = Number(range.value);
      out.textContent = '第 ' + layer + ' 层';
      render();
    });
    out.textContent = '第 ' + layer + ' 层';
    responsive(render);
  }

  // ---------------------------------------------------------------- 第 2 章：发票 → 框 → token
  function widgetInvoice() {
    var w = document.getElementById('w-invoice');
    if (!w || !D) return;
    var canvas = w.querySelector('.doc-canvas'), detail = w.querySelector('.detail'), strip = w.querySelector('.tok-strip');
    var btns = w.querySelectorAll('.seg-btns button');
    var PW = D.page[0], PH = D.page[1], VH = 520;
    var level = 'segment', sel = 2, selTok = -1;

    var svg = s('svg', { viewBox: '0 0 ' + PW + ' ' + VH, role: 'img', 'aria-label': '示例发票的上半页' }, canvas);
    s('rect', { x: 1, y: 1, width: PW - 2, height: VH - 2, rx: 6, style: 'fill:var(--surface);stroke:var(--border)' }, svg);
    txt(svg, PW - 12, VH - 12, '页面 850 × 1100 像素（只画上半页）', { 'text-anchor': 'end', 'class': 'd-muted', 'font-size': 15 });
    var rects = D.segments.map(function (sg, i) {
      var b = sg.box, g = s('g', null, svg);
      var r = s('rect', { x: b[0] - 3, y: b[1] - 3, width: b[2] - b[0] + 6, height: b[3] - b[1] + 6, rx: 3,
        'class': 'seg-rect', tabindex: 0, role: 'button', 'aria-label': sg.text }, g);
      var fs = Math.min(22, (b[3] - b[1]) * 0.7);
      txt(g, b[0] + 2, (b[1] + b[3]) / 2, sg.text, { 'class': 'seg-txt', 'font-size': fs, 'dominant-baseline': 'central' });
      r.addEventListener('click', function () { selTok = -1; select(i); });
      r.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); selTok = -1; select(i); }
      });
      return r;
    });
    var wordLayer = s('g', null, svg);
    D.words.forEach(function (wd) {
      var b = wd.wordbox;
      s('rect', { x: b[0] * PW / 1000, y: b[1] * PH / 1000, width: (b[2] - b[0]) * PW / 1000, height: (b[3] - b[1]) * PH / 1000,
        'class': 'd-dash', style: 'stroke:var(--layout);pointer-events:none' }, wordLayer);
    });

    strip.innerHTML = D.tokens.map(function (t, i) {
      return '<span data-i="' + i + '" class="' + (t.wid == null ? 'special' : '') + '" tabindex="0">' + esc(t.t) + '</span>';
    }).join('');
    var spans = strip.querySelectorAll('span');
    for (var k = 0; k < spans.length; k++) {
      (function (sp) {
        function pick() {
          var i = Number(sp.getAttribute('data-i')), t = D.tokens[i];
          selTok = i;
          if (t.wid != null) select(D.words[t.wid].seg); else select(-1);
        }
        sp.addEventListener('click', pick);
        sp.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(); } });
      })(spans[k]);
    }

    function labText(t) {
      if (t.lab == null) return '-100（特殊 token）';
      if (t.lab === '-100') return '-100（非首个）';
      return t.lab;
    }
    function select(i) {
      sel = i;
      rects.forEach(function (r, j) { r.classList.toggle('on', j === i); });
      wordLayer.style.display = level === 'word' ? '' : 'none';
      for (var k = 0; k < spans.length; k++) {
        var t = D.tokens[k];
        var on = i >= 0 ? (t.wid != null && D.words[t.wid].seg === i) : k === selTok;
        spans[k].classList.toggle('on', on);
      }
      if (i < 0) {
        var t0 = D.tokens[selTok];
        detail.innerHTML = '<p><b>' + esc(t0.t) + '</b> 是特殊 token，不属于任何词。</p><dl class="kv"><dt>bbox</dt><dd>' +
          JSON.stringify(t0.bbox) + '</dd><dt>标签</dt><dd>-100（不算 loss）</dd></dl>' +
          '<p style="font-size:13.5px;color:var(--muted);margin-top:10px">官方 FUNSD 代码给 &lt;s&gt; 和 &lt;/s&gt; 都填 [0,0,0,0]；论文写的是 &lt;/s&gt; 填 1000。</p>';
        return;
      }
      var sg = D.segments[i], b = sg.box, nb = sg.norm;
      var rows = [];
      D.tokens.forEach(function (t, j) {
        if (t.wid == null || D.words[t.wid].seg !== i) return;
        var box = level === 'word' ? D.words[t.wid].wordbox : t.bbox;
        rows.push('<tr' + (j === selTok ? ' style="background:var(--layout-wash)"' : '') + '><td><code>' + esc(t.t) + '</code></td><td>' +
          esc(D.words[t.wid].w) + '</td><td><code>[' + box.join(', ') + ']</code></td><td>' + labText(t) + '</td></tr>');
      });
      detail.innerHTML =
        '<p style="margin-bottom:6px"><b>' + esc(sg.text) + '</b> <span class="pill">' + sg.funsd + '</span></p>' +
        '<dl class="kv"><dt>像素框</dt><dd>(' + b.join(', ') + ')</dd>' +
        '<dt>归一化</dt><dd>[' + nb.join(', ') + ']</dd>' +
        '<dt>算法</dt><dd>⌊1000·x/850⌋, ⌊1000·y/1100⌋</dd></dl>' +
        '<div class="table-wrap" style="margin-top:10px"><table><thead><tr><th>token</th><th>词</th><th>' +
        (level === 'word' ? '词框（替代方案）' : '送进模型的 bbox') + '</th><th>标签</th></tr></thead><tbody>' + rows.join('') + '</tbody></table></div>';
    }
    for (var b = 0; b < btns.length; b++) {
      btns[b].addEventListener('click', function () {
        level = this.getAttribute('data-v');
        for (var j = 0; j < btns.length; j++) btns[j].classList.toggle('on', btns[j] === this);
        select(sel);
      });
    }
    select(sel);
  }

  // ---------------------------------------------------------------- 第 3 章：一个框的六次查表
  function widgetLookup() {
    var w = document.getElementById('w-lookup');
    if (!w || !D) return;
    var selEl = w.querySelector('select'), ins = w.querySelectorAll('input[type=number]');
    var grid = w.querySelector('.lookup-grid'), msg = w.querySelector('.lookup-msg');
    var presets = D.segments.map(function (sg) { return { name: sg.text, box: sg.norm }; });
    presets.push({ name: '<s> 和 </s>（官方代码）', box: [0, 0, 0, 0] });
    presets.push({ name: '</s>（论文写法）', box: [1000, 1000, 1000, 1000] });
    presets.push({ name: '坏框：x1 < x0', box: [500, 100, 300, 120] });
    presets.push({ name: '坏框：没归一化的像素坐标', box: [660, 460, 1240, 1480] });
    selEl.innerHTML = presets.map(function (p, i) { return '<option value="' + i + '">' + esc(p.name) + '</option>'; }).join('');
    function load(i) {
      var bx = presets[i].box;
      for (var k = 0; k < 4; k++) ins[k].value = bx[k];
      update();
    }
    function update() {
      var v = [];
      for (var k = 0; k < 4; k++) v.push(parseInt(ins[k].value, 10) || 0);
      var cells = [
        ['左 x0', 'x 表', v[0]], ['上 y0', 'y 表', v[1]], ['右 x1', 'x 表', v[2]],
        ['下 y1', 'y 表', v[3]], ['高 y1−y0', 'h 表', v[3] - v[1]], ['宽 x1−x0', 'w 表', v[2] - v[0]]
      ];
      var bad = [];
      grid.innerHTML = cells.map(function (c) {
        var ok = c[2] >= 0 && c[2] <= 1023;
        if (!ok) bad.push(c[0] + ' = ' + c[2]);
        return '<div style="' + (ok ? '' : 'background:transparent;outline:2px solid var(--bad)') + '">' + c[0] +
          '<b>' + c[2] + '</b>查 ' + c[1] + '</div>';
      }).join('');
      if (bad.length) {
        msg.innerHTML = '<b style="color:var(--bad)">IndexError</b>：' + esc(bad.join('，')) +
          ' 超出了 0..1023。坐标表只有 1024 行，负数或大于 1023 的索引会直接报错（或在 GPU 上触发 device-side assert）。';
      } else {
        msg.innerHTML = '6 次查表，每次得到 128 维 → 拼成 <span class="chip l">768</span> → 线性层 → <span class="chip l">192</span>' +
          ' → 加上 1D 位置向量 <span class="chip l">192</span> → LayerNorm → <span class="chip l">192</span>';
      }
    }
    selEl.addEventListener('change', function () { load(Number(selEl.value)); });
    for (var k = 0; k < 4; k++) ins[k].addEventListener('input', update);
    load(2);
    selEl.value = '2';
  }

  // ---------------------------------------------------------------- 第 5 章：KPL 的 7×7 区域
  function widgetKpl() {
    var w = document.getElementById('w-kpl');
    if (!w || !D) return;
    var host = w.querySelector('.kpl-wrap'), out = w.querySelector('.kpl-out');
    var N = 7;
    function region(x, y) {
      var col = Math.min(N - 1, Math.floor(x * N / 1001)), row = Math.min(N - 1, Math.floor(y * N / 1001));
      return row * N + col;
    }
    var svg = s('svg', { viewBox: '-4 -4 1008 1008', role: 'img', 'aria-label': '归一化后的页面被切成 7×7 个区域' }, host);
    var cells = [];
    var cw = 1001 / N;
    for (var r = 0; r < N; r++) {
      for (var c = 0; c < N; c++) {
        cells.push(s('rect', { x: c * cw, y: r * cw, width: cw, height: cw, 'class': 'kpl-cell' }, svg));
        txt(svg, c * cw + 10, r * cw + 34, String(r * N + c), { 'class': 'kpl-id', 'font-size': 30 });
      }
    }
    var boxes = D.segments.map(function (sg, i) {
      var b = sg.norm;
      var el = s('rect', { x: b[0], y: b[1], width: b[2] - b[0], height: b[3] - b[1], 'class': 'kpl-box', tabindex: 0,
        role: 'button', 'aria-label': sg.text }, svg);
      el.addEventListener('click', function () { pick(i); });
      el.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(i); } });
      return el;
    });
    var ptLayer = s('g', null, svg);
    function pick(i) {
      var b = D.segments[i].norm;
      var pts = [
        ['左上', b[0], b[1], 'hit-tl'],
        ['右下', b[2], b[3], 'hit-br'],
        ['中心', Math.floor((b[0] + b[2]) / 2), Math.floor((b[1] + b[3]) / 2), 'hit-ct']
      ];
      cells.forEach(function (c) { c.setAttribute('class', 'kpl-cell'); });
      boxes.forEach(function (bx, j) { bx.classList.toggle('on', j === i); });
      ptLayer.innerHTML = '';
      var lines = pts.map(function (p) {
        var id = region(p[1], p[2]);
        cells[id].setAttribute('class', 'kpl-cell ' + p[3]);
        s('circle', { cx: p[1], cy: p[2], r: 9, 'class': 'kpl-pt', style: 'fill:var(--layout)' }, ptLayer);
        return p[0] + ' (' + p[1] + ', ' + p[2] + ') → 区域 <b>' + id + '</b>（第 ' + Math.floor(id / N) + ' 行第 ' + (id % N) + ' 列）';
      });
      out.innerHTML = '<b>' + esc(D.segments[i].text) + '</b>　框 [' + b.join(', ') + ']<br>' + lines.join('<br>') +
        '<br><span style="color:var(--muted)">模型要做的是三个 49 类分类：分别猜这三个点落在哪个区域。</span>';
    }
    pick(13);
  }

  // ---------------------------------------------------------------- 启动
  buildNav();
  buildPager();
  highlightAll();
  chartRobertaCos();
  chartCoord();
  chartRatio();
  widgetAttn();
  widgetInvoice();
  widgetLookup();
  widgetKpl();
})();
