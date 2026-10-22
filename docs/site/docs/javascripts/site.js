/* Shared site controls and contextual help. */
(function () {
  "use strict";

  /* The maintainer's text slots (docs/content/site-copy.yml), which a page
     carries as `data-site-copy`, read by "group.key". A blank slot is "";
     PLACEHOLDER reads as the slot's own key, so an unwritten slot shows. */
  var siteCopyGroups = null;
  function siteCopy(key) {
    if (!siteCopyGroups) {
      siteCopyGroups = {};
      document.querySelectorAll('[data-site-copy]').forEach(function (host) {
        var parsed = JSON.parse(host.dataset.siteCopy);
        Object.keys(parsed).forEach(function (group) {
          siteCopyGroups[group] = Object.assign(siteCopyGroups[group] || {}, parsed[group]);
        });
      });
    }
    var parts = key.split('.');
    var value = (siteCopyGroups[parts[0]] || {})[parts[1]] || '';
    return value === 'PLACEHOLDER' ? '\u27e8' + key + '\u27e9' : value;
  }
  /* Markdown links in a slot: [text](url). A link to the public site address
     resolves against this build's root, so it works on every build. */
  var LINK = /\[([^\]\n]+)\]\(([^)\s]+)\)/g;
  var PUBLIC_SITE = 'https://saminulh.github.io/dpstats/';
  var siteRoot = document.currentScript ? new URL('../', document.currentScript.src).href : '/';
  function linkable(url) { return !/^[A-Za-z][\w+.-]*:/.test(url) || /^https?:\/\//.test(url); }
  function siteHref(url) { return url.indexOf(PUBLIC_SITE) === 0 ? siteRoot + url.slice(PUBLIC_SITE.length) : url; }
  /* A slot's `{name}` fields take the values the caller passes, as text;
     `{{` and `}}` are literal braces. A field the caller does not pass throws
     rather than show its raw name. Fields fill after links are found, so a
     value never becomes a link. */
  var FIELD = /\{\{|\}\}|\{([A-Za-z_]\w*)\}/g;
  function fillFields(key, text, fields) {
    return text.replace(FIELD, function (whole, name) {
      if (!name) return whole[0];
      if (!fields || !Object.prototype.hasOwnProperty.call(fields, name)) {
        throw new Error('site copy ' + key + ': no value for field {' + name + '}');
      }
      return String(fields[name]);
    });
  }
  /* One slot as plain text, for tooltips and attributes: links keep their text. */
  function siteCopyText(key, fields) {
    var text = siteCopy(key);
    var pieces = [], last = 0, match;
    LINK.lastIndex = 0;
    while ((match = LINK.exec(text))) {
      if (!linkable(match[2])) continue;
      pieces.push(fillFields(key, text.slice(last, match.index), fields), fillFields(key, match[1], fields));
      last = LINK.lastIndex;
    }
    pieces.push(fillFields(key, text.slice(last), fields));
    return pieces.join('');
  }
  /* Write one slot into *node*, marked so a placeholder is styled as one. */
  function fillSiteCopy(node, key, fields) {
    var text = siteCopy(key);
    var parts = [], last = 0, match;
    LINK.lastIndex = 0;
    while ((match = LINK.exec(text))) {
      if (!linkable(match[2])) continue;
      parts.push(fillFields(key, text.slice(last, match.index), fields));
      var a = document.createElement('a');
      a.href = siteHref(match[2]);
      a.textContent = fillFields(key, match[1], fields);
      parts.push(a);
      last = LINK.lastIndex;
    }
    parts.push(fillFields(key, text.slice(last), fields));
    node.replaceChildren.apply(node, parts);
    node.dataset.copySlot = key;
    node.classList.add('dp-authored-copy');
    node.classList.toggle('dp-copy-slot', text === '\u27e8' + key + '\u27e9');
    return text;
  }
  /* One slot in a fresh span, for an element that also shows other text or
     must keep its own display: the slot's marking stays on the span. */
  function siteCopySpan(key, fields) {
    var span = document.createElement('span');
    fillSiteCopy(span, key, fields);
    return span;
  }
  /* One slot as a token list for a page that builds its own pieces: text
     runs, and in each `{name}` field's place the caller's token (a reader
     term, or a name the page fills). An unwritten slot is its marker alone. */
  function siteCopyTokens(key, fields) {
    var text = siteCopy(key);
    var tokens = [], last = 0, match;
    var field = new RegExp(FIELD.source, 'g');
    while ((match = field.exec(text))) {
      tokens.push(text.slice(last, match.index));
      last = field.lastIndex;
      if (!match[1]) { tokens.push(match[0][0]); continue; }
      if (!fields || !Object.prototype.hasOwnProperty.call(fields, match[1])) {
        throw new Error('site copy ' + key + ': no value for field {' + match[1] + '}');
      }
      tokens.push(fields[match[1]]);
    }
    tokens.push(text.slice(last));
    return tokens.filter(function (token) { return token !== ''; });
  }
  /* Tabler Icons 3.35.0 (MIT) the scripts place: the funnel every filter
     button carries. */
  window.dpIcons = {
    filter: '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="icon" aria-hidden="true"><path d="M4 4h16v2.172a2 2 0 0 1 -.586 1.414l-4.414 4.414v7l-6 2v-8.5l-4.48 -4.928a2 2 0 0 1 -.52 -1.345v-2.227z"/></svg>',
  };
  window.dpSiteCopy = siteCopyText;
  window.dpSiteCopyTokens = siteCopyTokens;
  window.dpSiteCopyFill = fillSiteCopy;
  window.dpSiteCopySpan = siteCopySpan;

  var themeButton = document.getElementById('theme-toggle');
  function setTheme(value) {
    document.documentElement.dataset.bsTheme = value;
    document.body.dataset.bsTheme = value;
    // The icon shows the mode a press switches to; the name says so in words.
    var label = siteCopyText(value === 'dark' ? 'site.theme_light' : 'site.theme_dark');
    themeButton.setAttribute('aria-label', label);
    themeButton.title = label;
  }
  if (themeButton) {
    setTheme(document.documentElement.dataset.bsTheme || 'dark');
    // Only a reader's own click is remembered; an unchosen page stays on the default.
    themeButton.addEventListener('click', function () {
      var value = document.documentElement.dataset.bsTheme === 'dark' ? 'light' : 'dark';
      setTheme(value);
      try { localStorage.setItem('dp-theme-choice', value); } catch (_) {}
    });
  }
  var sidebar = document.getElementById('section-sidebar');
  var sidebarToggle = document.getElementById('sidebar-toggle');
  if (sidebar && sidebarToggle) {
    var desktop = window.matchMedia('(min-width: 992px)');
    function sidebarLabel() {
      var expanded = desktop.matches ? !document.documentElement.hasAttribute('data-sidebar-collapsed') : sidebar.classList.contains('show');
      sidebarToggle.title = siteCopyText(expanded ? 'site.sidebar_collapse' : 'site.sidebar_expand');
      sidebarToggle.setAttribute('aria-label', sidebarToggle.title);
      sidebarToggle.setAttribute('aria-expanded', String(expanded));
      // In the collapsed rail a link shows only its icon; its title is the tooltip.
      sidebar.querySelectorAll('.nav-link').forEach(function (link) {
        var title = link.querySelector('.nav-link-title');
        if (!title) return;
        if (desktop.matches && !expanded) link.title = title.textContent.trim();
        else link.removeAttribute('title');
      });
    }
    sidebarToggle.addEventListener('click', function () {
      if (!desktop.matches) {
        tabler.Offcanvas.getOrCreateInstance(sidebar).toggle();
        return;
      }
      var collapsed = document.documentElement.toggleAttribute('data-sidebar-collapsed');
      try { sessionStorage.setItem('dp-sidebar-collapsed', String(collapsed)); } catch (_) {}
      sidebarLabel();
      requestAnimationFrame(function () {
        document.querySelectorAll('.js-plotly-plot').forEach(function (plot) {
          if (plot.getBoundingClientRect().width && plot.getBoundingClientRect().height) window.Plotly.Plots.resize(plot);
        });
      });
    });
    sidebar.addEventListener('shown.bs.offcanvas', sidebarLabel);
    sidebar.addEventListener('hidden.bs.offcanvas', function () {
      sidebarLabel();
      if (!desktop.matches) sidebarToggle.focus();
    });
    desktop.addEventListener('change', function () {
      if (desktop.matches) {
        var drawer = tabler.Offcanvas.getInstance(sidebar);
        if (drawer) drawer.hide();
      }
      sidebarLabel();
    });
    sidebarLabel();
    // The page's own link starts in view, a third of the way down the sidebar.
    var here = sidebar.querySelector('.nav-link.active');
    if (here && desktop.matches && sidebar.scrollHeight > sidebar.clientHeight) {
      sidebar.scrollTop = here.getBoundingClientRect().top - sidebar.getBoundingClientRect().top - sidebar.clientHeight / 3;
    }
  }
  // Clipboard interaction progressively enhances ordinary rendered code: a
  // corner button that stays faint until the block is hovered or focused.
  // API signatures are read, not pasted, so they get none.
  // Notebook input cells render a bare <pre>; their outputs are results, not code.
  document.querySelectorAll('pre:has(code), .highlight-ipynb > pre').forEach(function (pre) {
    if (pre.closest('.doc-signature')) return;
    // The wrapper sits outside the scrolling block so the button stays put
    // when a long line scrolls sideways.
    var block = pre.closest('.highlight, .highlight-ipynb') || pre;
    var host = document.createElement('div');
    host.className = 'dp-code';
    block.before(host);
    host.appendChild(block);
    var button = document.createElement('button');
    button.type = 'button';
    button.className = 'dp-copy';
    button.setAttribute('aria-label', siteCopyText('site.copy_code'));
    button.innerHTML = '<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2"/></svg><span class="dp-copy-status" aria-live="polite"></span>';
    var status = button.querySelector('.dp-copy-status');
    var reset;
    var source = pre.querySelector('code') || pre;
    // Where the page may not write the clipboard, copying the selected block
    // still works; when neither does, the block is left selected to copy by hand.
    function copySelection() {
      var range = document.createRange();
      range.selectNodeContents(source);
      var selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      var copied = document.execCommand('copy');
      if (copied) selection.removeAllRanges();
      return copied;
    }
    button.addEventListener('click', async function () {
      try {
        await navigator.clipboard.writeText(source.textContent);
        status.replaceChildren(siteCopySpan('site.copied'));
      } catch (_) {
        if (copySelection()) status.replaceChildren(siteCopySpan('site.copied'));
        else status.replaceChildren(window.dpSiteCopySpan('site.copy_selected_hint'));
      }
      clearTimeout(reset);
      reset = setTimeout(function () { status.textContent = ''; }, 1600);
    });
    host.appendChild(button);
  });
  // The rail marks the last heading scrolled past the top quarter of the view,
  // or the last one once the page cannot scroll further.
  var rail = document.querySelector('.dp-toc');
  if (rail) {
    var entries = Array.prototype.map.call(rail.querySelectorAll('a[href^="#"]'), function (link) {
      return { link: link, heading: document.getElementById(decodeURIComponent(link.hash.slice(1))) };
    }).filter(function (entry) { return entry.heading; });
    var current = null;
    var chosen = null;
    entries.forEach(function (entry) {
      entry.link.addEventListener('click', function () { chosen = entry; mark(); });
    });
    var mark = function () {
      var line = window.innerHeight / 4;
      var active = entries[0];
      entries.forEach(function (entry) {
        if (entry.heading.getBoundingClientRect().top <= line) active = entry;
      });
      // At the page foot the last headings can never reach the line.
      // A page too short to scroll shows every heading, so the first stays marked.
      if (window.scrollY > 0 &&
          window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 2) {
        active = entries[entries.length - 1];
      }
      // A heading the reader just chose stays marked while it is in view.
      if (chosen) {
        var box = chosen.heading.getBoundingClientRect();
        if (box.bottom > 0 && box.top < window.innerHeight) active = chosen;
        else chosen = null;
      }
      if (!active || active === current) return;
      if (current) current.link.removeAttribute('aria-current');
      active.link.setAttribute('aria-current', 'true');
      current = active;
    };
    window.addEventListener('scroll', mark, { passive: true });
    mark();
  }
  // A list that scrolls within the page fades at its foot while rows remain below.
  function hintMore(list) {
    // Reads layout, so it waits for the frame: a read per DOM change would
    // lay out the page after every write a list re-render makes.
    var pending = null;
    function mark() {
      if (pending !== null) return;
      pending = requestAnimationFrame(function () {
        pending = null;
        list.classList.toggle('dp-more-below', list.scrollTop + list.clientHeight < list.scrollHeight - 2);
      });
    }
    list.addEventListener('scroll', mark, { passive: true });
    new ResizeObserver(mark).observe(list);
    new MutationObserver(mark).observe(list, { childList: true, subtree: true });
    mark();
  }
  window.dpHintMore = hintMore;
  document.querySelectorAll('.dp-plotted-rows, .dp-home .dp-table-scroll').forEach(hintMore);
  var search = document.getElementById('search-panel');
  // "/" opens search from anywhere but a field the reader is typing in.
  if (search) document.addEventListener('keydown', function (event) {
    if (event.key !== '/' || event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.target.closest('input, textarea, select, [contenteditable]')) return;
    event.preventDefault();
    tabler.Offcanvas.getOrCreateInstance(search).show();
  });
  // Pagefind's ranking for every query (defaults 0.75, 1.0, 1.4): page length and
  // saturation lowered, so a few long authored pages are not crowded out by
  // hundreds of short generated ones.
  var SEARCH_RANKING = { pageLength: 0.2, termFrequency: 1.0, termSaturation: 1.0 };
  var searchLoaded = null;
  // The Pagefind interface loads when search first opens, so no other page
  // view pays for it. Its elements wait in a template until the instance has
  // its ranking: an element creates the instance as soon as it connects.
  function loadSearch() {
    if (searchLoaded) return searchLoaded;
    var ui = document.getElementById('dp-search-ui');
    searchLoaded = new Promise(function (resolve) {
      var style = document.createElement('link');
      style.rel = 'stylesheet';
      style.href = ui.dataset.href;
      document.head.appendChild(style);
      var script = document.createElement('script');
      script.src = ui.dataset.src;
      script.onload = function () {
        // The bundle sits beside the script; a site served under a subpath
        // has no /pagefind/ at its host's root.
        var bundlePath = new URL('.', script.src).pathname;
        window.PagefindComponents.configureInstance('default', { bundlePath: bundlePath, ranking: SEARCH_RANKING });
        ui.replaceWith(ui.content.cloneNode(true));
        resolve(true);
      };
      script.onerror = function () {
        var failed = document.createElement('p');
        failed.setAttribute('role', 'alert');
        window.dpSiteCopyFill(failed, 'site.search_load_error');
        ui.after(failed);
        resolve(false);
      };
      document.head.appendChild(script);
    });
    return searchLoaded;
  }
  if (search) search.addEventListener('show.bs.offcanvas', loadSearch);
  if (search) search.addEventListener('shown.bs.offcanvas', function () {
    loadSearch().then(function (loaded) {
      var field = loaded && search.querySelector('pagefind-input input');
      if (field && search.classList.contains('show')) field.focus();
    });
  });
  if (search) {
    var searchTrigger = document.querySelector('[data-bs-target="#search-panel"]');
    // Escape closes the panel wherever focus sits, even on BODY, and the
    // trigger gets focus back.
    document.addEventListener('keydown', function (event) {
      if (event.key !== 'Escape' || !search.classList.contains('show')) return;
      tabler.Offcanvas.getOrCreateInstance(search).hide();
    });
    search.addEventListener('hidden.bs.offcanvas', function () {
      if (searchTrigger) searchTrigger.focus();
    });
  }
  // A link to Home's #task-menu opens the global task picker.
  function openTasks() {
    if (location.hash !== '#task-menu') return;
    var menu = document.getElementById('global-menu');
    var picker = document.getElementById('task-picker');
    if (!picker) return;
    function showPicker() { tabler.Dropdown.getOrCreateInstance(picker).show(); }
    if (getComputedStyle(menu).display === 'none') {
      menu.addEventListener('shown.bs.collapse', showPicker, { once: true });
      tabler.Collapse.getOrCreateInstance(menu, { toggle: false }).show();
    } else showPicker();
  }
  openTasks();
  window.addEventListener('hashchange', openTasks);
  // Native non-modal popovers supply touch, keyboard, Escape and light-dismiss
  // behavior for approved contextual help. They do not block the page. The
  // trigger is the control's own label: pointing at it or tabbing to it opens
  // the help, which stays while the pointer rests on either one.
  /* The one reader-facing number format: three significant figures with digit
     grouping and a true minus sign, and from a million up a power of ten
     written as 10⁷ or 1.2 × 10⁷, which Intl cannot write, never as e-notation. */
  var SUPERSCRIPTS = '⁰¹²³⁴⁵⁶⁷⁸⁹';
  var THREE_FIGURES = new Intl.NumberFormat(undefined, { maximumSignificantDigits: 3 });
  window.dpNumber = function (value) {
    if (!Number.isFinite(value)) return Number.isNaN(value) ? String(value) : (value > 0 ? '∞' : '−∞');
    var rounded = Number(value.toPrecision(3));
    if (Math.abs(rounded) < 1e6) return THREE_FIGURES.format(value).replace('-', '−');
    var exponent = Math.floor(Math.log10(Math.abs(rounded)));
    var mantissa = Number((rounded / Math.pow(10, exponent)).toPrecision(3));
    if (Math.abs(mantissa) >= 10) { mantissa /= 10; exponent += 1; }
    var power = '10' + String(exponent).replace(/\d/g, function (digit) { return SUPERSCRIPTS[digit]; });
    return (mantissa === 1 ? power : String(mantissa).replace(/^-/, '−') + ' × ' + power);
  };
  /* Reads back what dpNumber writes (10⁷, 1.2 × 10⁷, 1,030) and plain or
     e-notation typing; NaN for anything else. */
  window.dpParseNumber = function (text) {
    var written = String(text).trim().replace(/,/g, '').replace(/^−/, '-');
    var power = written.match(/^(?:([-+]?\d*\.?\d+)\s*[×x]\s*)?10([⁻]?[⁰-⁹]+)$/);
    if (power) {
      var exponent = Number(power[2].replace('⁻', '-').replace(/[⁰-⁹]/g, function (digit) {
        return String(SUPERSCRIPTS.indexOf(digit));
      }));
      return (power[1] === undefined ? 1 : Number(power[1])) * Math.pow(10, exponent);
    }
    return written === '' ? NaN : Number(written);
  };
  window.dpContextualHelp = function (trigger, content) {
    // A link keeps its own click, which follows it; a button's click only
    // ever shows the help, since pointing already opened it and a toggle
    // would close the help the reader just asked for.
    if (trigger.tagName !== 'A') {
      trigger.type = 'button';
      trigger.setAttribute('popovertarget', content.id);
      trigger.setAttribute('popovertargetaction', 'show');
    }
    content.className = 'dp-context-help';
    content.removeAttribute('role');
    content.setAttribute('popover', 'auto');
    trigger.after(content);
    var leaving;
    var dismissed = false;
    function open() {
      clearTimeout(leaving);
      if (dismissed || !content.textContent.trim() || content.matches(':popover-open')) return;
      content.showPopover();
    }
    function leave() {
      clearTimeout(leaving);
      leaving = setTimeout(function () {
        if (trigger.matches(':hover, :focus') || content.matches(':hover, :focus-within')) return;
        if (content.matches(':popover-open')) content.hidePopover();
      }, 200);
    }
    function release() {
      dismissed = false;
      leave();
    }
    // Escape dismisses the help without moving the pointer or losing the
    // control's focus; pointing or focusing it again brings the help back.
    document.addEventListener('keydown', function (event) {
      if (event.key !== 'Escape' || !content.matches(':popover-open')) return;
      dismissed = true;
      clearTimeout(leaving);
      content.hidePopover();
    });
    [trigger, content].forEach(function (element) {
      element.addEventListener('mouseenter', open);
      element.addEventListener('mouseleave', release);
    });
    trigger.addEventListener('focus', open);
    trigger.addEventListener('blur', release);
    // A link inside the help is reachable by Tab; leaving it closes the help.
    content.addEventListener('focusout', release);
    content.addEventListener('toggle', function (event) {
      if (event.newState !== 'open') return;
      var rect = trigger.getBoundingClientRect();
      content.style.left = Math.max(12, Math.min(rect.left, innerWidth - content.offsetWidth - 12)) + 'px';
      // Below the trigger, or above it when only that fits, so the help never
      // covers the link it opened from.
      var height = content.offsetHeight;
      var above = rect.top - height - 6;
      var fitsBelow = rect.bottom + 6 + height + 12 <= innerHeight;
      var top = fitsBelow || above < 12 ? Math.min(rect.bottom + 6, innerHeight - height - 12) : above;
      content.style.top = Math.max(12, top) + 'px';
    });
  };
  // A defined term the page rendered with its help beside it, such as a task
  // table's "Privacy interpretation" label.
  document.querySelectorAll('[data-dp-term]').forEach(function (trigger) {
    var content = document.getElementById(trigger.getAttribute('data-dp-term'));
    if (!content) return;
    window.dpContextualHelp(trigger, content);
    content.hidden = false;
  });
  /* One instance's card: its label linking its page, its study, and the
     facts a reader compares instances on. Each task's cards sit in one file
     beside that task's instance pages, read once on first use. */
  var instanceCards = {};
  var CARD_FACTS = [
    ['field', 'site.fact_field'],
    ['n', 'site.card_samples'],
    ['d', 'site.card_dimension'],
    ['kappa', 'site.fact_condition_number'],
  ];
  window.dpInstanceCard = function (href) {
    var page = new URL(href, document.baseURI);
    var uuid = page.pathname.replace(/\/$/, '').split('/').pop();
    var file = new URL('../../data/instances.json', page).href;
    instanceCards[file] = instanceCards[file] || fetch(file).then(function (response) {
      if (!response.ok) throw new Error(response.status + ' ' + file);
      return response.json();
    });
    return instanceCards[file].then(function (cards) {
      var card = cards[uuid];
      if (!card) throw new Error('No card for ' + uuid);
      var box = document.createElement('div');
      box.className = 'dp-instance-card';
      var title = document.createElement('a');
      title.href = page.href;
      title.className = 'dp-instance-card-title';
      var label = document.createElement('code');
      label.textContent = card.label;
      title.appendChild(label);
      var study = document.createElement('div');
      study.className = 'text-secondary';
      study.textContent = card.study;
      var facts = document.createElement('dl');
      // Each row: its term, its value, whether the term is code, whether it is a slot.
      var rows = CARD_FACTS.filter(function (fact) { return card[fact[0]] !== null && card[fact[0]] !== ''; })
        .map(function (fact) {
          var value = card[fact[0]];
          return [fact[1], typeof value === 'number' ? window.dpNumber(value) : value, false, true];
        });
      // The headline baseline is named as the site names it everywhere: in code.
      if (card.baseline) rows.push([card.baseline[0], window.dpNumber(card.baseline[1]), true]);
      rows.forEach(function (row) {
        var term = document.createElement('dt');
        if (row[2]) {
          var name = document.createElement('code');
          name.textContent = row[0];
          term.appendChild(name);
        } else if (row[3]) fillSiteCopy(term, row[0]);
        else term.textContent = row[0];
        var value = document.createElement('dd');
        value.textContent = row[1];
        facts.append(term, value);
      });
      box.append(title, study, facts);
      return box;
    });
  };
  // An instance link previews the instance's card on hover or focus and still
  // navigates on click; `id` names the card's content element.
  window.dpInstancePreview = function (link, id) {
    var content = document.createElement('div');
    content.id = id;
    var requested = false;
    function load() {
      if (requested) return;
      requested = true;
      window.dpInstanceCard(link.href).then(function (card) {
        content.replaceChildren(card);
        // The card arrived while the reader still points at or focuses the link.
        if (link.matches(':hover')) link.dispatchEvent(new Event('mouseenter'));
        else if (link.matches(':focus')) link.dispatchEvent(new Event('focus'));
      }).catch(function () { requested = false; });
    }
    link.addEventListener('mouseenter', load);
    link.addEventListener('focus', load);
    window.dpContextualHelp(link, content);
  };
  // A table of instances previews each one's card from its link.
  document.querySelectorAll('[data-dp-instance-cards] a[href*="/instances/"]').forEach(function (link, index) {
    window.dpInstancePreview(link, 'dp-instance-card-' + index);
  });
  // A code block or table wider than its column scrolls; keyboard readers
  // reach it by Tab so the arrow keys can scroll it, and only while it does.
  function scrollStop(box) {
    if (box.scrollWidth > box.clientWidth + 1) box.tabIndex = 0;
    else box.removeAttribute('tabindex');
  }
  var resized = new ResizeObserver(function (entries) {
    entries.forEach(function (entry) { scrollStop(entry.target.closest('pre, .dp-table-scroll') || entry.target); });
  });
  document.querySelectorAll('.dp-content pre, .dp-content .dp-table-scroll').forEach(function (box) {
    scrollStop(box);
    resized.observe(box);
    if (box.firstElementChild) resized.observe(box.firstElementChild);
  });
})();
