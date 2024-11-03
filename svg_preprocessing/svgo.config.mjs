export default {
  multipass: true, // boolean
  datauri: 'unenc', // 'base64'|'enc'|'unenc'
  js2svg: {
    indent: 4, // number
    pretty: true, // boolean
  },
  plugins: [
    "moveGroupAttrsToElems",
    "collapseGroups",
    "mergePaths"
    {
      name: "convertPathData",
      params: {
        applyTransforms: true,
        applyTransformsStroked: true,
        straightCurves: false,
        convertToQ: false,
        lineShorthands: false,
        convertToZ: false,
        curveSmoothShorthands: false,
        floatPrecision: 2,
        transformPrecision: 2,
        smartArcRounding: false,
        removeUseless: true,
        collapseRepeated: true,
        utilizeAbsolute: false,
        negativeExtraSpace: true,
        forceAbsolutePath: false
      }
    },
    {
      name: "convertColors",
      params: {
        currentColor: false,
        names2hex: true,
        rgb2hex: true,
        convertCase: "lower",
        shorthex: false,
        shortname: false
      }
    },
    "cleanupIds"
  ],
};