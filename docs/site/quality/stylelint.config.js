// What the site stylesheet may write directly. Colours and type sizes come
// from tokens: custom properties declared on :root or a theme, the one place a
// literal colour is written (stylesheet.mjs leaves those out). Findings that exist today are listed in
// stylesheet-baseline.json and may only shrink (README.md, "Limits").
export default {
  plugins: ["stylelint-declaration-strict-value"],
  ignoreFiles: ["**/vendor/**"],
  rules: {
    "color-no-hex": true,
    "color-named": "never",
    "function-disallowed-list": ["rgb", "rgba", "hsl", "hsla"],
    "declaration-property-value-allowed-list": {
      "font-size": ["/^var\\(--dp-size-/", "inherit", "0"],
    },
    "scale-unlimited/declaration-strict-value": [
      ["font-family"],
      { ignoreValues: ["inherit", "currentColor", "currentcolor", "transparent", "none", "initial", "unset"], disableFix: true },
    ],
  },
};
