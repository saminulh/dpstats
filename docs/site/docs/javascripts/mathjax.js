window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true
  },
  startup: {
    // nbconvert keeps notebook math delimiters in its Markdown output.
    elements: [".arithmatex", ".jp-MarkdownOutput"]
  }
};
