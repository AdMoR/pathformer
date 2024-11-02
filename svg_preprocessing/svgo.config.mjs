export default {
  multipass: true, // boolean
  datauri: 'unenc', // 'base64'|'enc'|'unenc'
  js2svg: {
    indent: 4, // number
    pretty: true, // boolean
  },
  plugins: [
    "moveGroupAttrsToElems",
    "convertStyleToAttrs",
    {
      name: "convertPathData",
      params: {
        applyTransforms: true,
        applyTransformsStroked: true,
        straightCurves: true,
        convertToQ: true,
        lineShorthands: true,
        convertToZ: true,
        curveSmoothShorthands: true,
        floatPrecision: 1,
        transformPrecision: 2,
        smartArcRounding: true,
        removeUseless: true,
        collapseRepeated: true,
        utilizeAbsolute: true,
        negativeExtraSpace: true,
        forceAbsolutePath: false
      }
    },
    "convertTransform",
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
