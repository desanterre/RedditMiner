const fs = require('fs');
const path = require('path');

const lines = [
"⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⣤⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
"⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⡏⠀⠀⠈⡆⠀⣠⠞⢻⡀⠀⠀⠀⠀⠀⠀⠀",
"⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⢷⣶⣤⠠⠓⠋⠁⠀⢸⡇⠀⠀⠀⠀⠀⠀⠀",
"⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠁⣀⣀⡀⠀⢀⣾⠃⠀⠀⠀⠀⠀⠀⠀",
"⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣤⡤⠖⠛⠉⠉⠉⠉⠉⠙⠛⢀⣀⣀⡀⠀⠀",
"⠀⠀⠀⠀⠀⠀⢀⡾⡽⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠀⠀⠉⠳⡄⠀⠀",
"⠀⠀⠀⠀⠀⠀⠸⣿⢡⣾⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣷⠀⠀",
"⠀⠀⠀⠀⠀⠀⢰⠇⠘⠛⠃⠀⠀⠀⠀⠒⠒⠢⠀⠀⠀⠠⡤⢄⡼⠃⠀⠀",
"⠀⠀⠀⠀⠀⠀⠸⡇⠀⠀⢀⠀⠀⠀⢀⠀⠀⠀⠀⠀⠀⠀⠘⢻⠀⠀⠀⠀",
"⠀⠀⠀⠀⠀⠀⠀⠻⣄⠀⠀⠈⠉⠉⠉⠀⠀⣴⠛⣦⠀⠀⢀⠞⠀⠀⠀⠀",
"⠀⠀⠀⠀⠀⠀⠀⠀⠈⠳⣤⣀⡀⠀⠀⠀⠀⡸⠀⠈⠛⣶⠋⠀⠀⠀⠀⠀",
"⠀⠀⠀⠀⠀⠀⠀⠀⡴⢛⡝⠁⠙⠿⡿⠛⠚⢇⠀⠀⠀⣽⠀⠀⠀⠀⠀⠀",
"⠀⠀⠀⠀⠀⠀⠀ ⢸⠁⣾⠁⠀⠀⠀⠀⠀⠀⢈⣳⡦⠚⠁⠀⠀⠀⠀⠀⠀",
"⠀⠀⠀⠀⠀⠀  ⠉⠻⣖⠦⣤⣤⡄⠀⢇⠀⠀⠀⠀⠀⠀⠀⠀⠀",
"⠀⠀⠀  ⠀   ⢰⠋⠀⠸⣿⡗⠀⠀⡼⠀⠀⠀⠀⠀⠀⠀⠀⠀",
"⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠓⠒⠉⠘⠧⡤⠴⠃Reddit Miner"
];

// Escape special XML characters for valid SVG
function escapeXml(unsafe) {
  return unsafe.replace(/[<>&'"]/g, function (c) {
    switch (c) {
      case '<': return '&lt;';
      case '>': return '&gt;';
      case '&': return '&amp;';
      case '\'': return '&apos;';
      case '"': return '&quot;';
    }
  });
}

const width = 700;
const height = 500;
const fontSize = 16;
const lineHeight = 20;
const startY = 20;
const centerX = width / 2;

let svgLines = '';
for (let i = 0; i < lines.length; i++) {
  const y = startY + i * lineHeight;
  const delay = (i * 0.15).toFixed(2);

  svgLines += `
    <text
      x="${centerX}" y="${y}"
      font-family="monospace"
      font-size="${fontSize}"
      fill="#FF4500"
      text-anchor="middle"
      dominant-baseline="middle"
      filter="url(#neonGlow)"
    >
      ${escapeXml(lines[i])}
      <animate attributeName="opacity" values="1;0.6;1" dur="2s" begin="${delay}s" repeatCount="indefinite" />
    </text>
  `;
}

const svgContent = `<?xml version="1.0" encoding="UTF-8"?>
<svg
  xmlns="http://www.w3.org/2000/svg"
  width="${width}"
  height="${height}"
  viewBox="0 0 ${width} ${height}"
>
  <defs>
    <filter id="neonGlow" x="-50%" y="-50%" width="200%" height="200%" color-interpolation-filters="sRGB">
      <feDropShadow dx="0" dy="0" stdDeviation="2" flood-color="#FF5700" flood-opacity="0.7"/>
      <feDropShadow dx="0" dy="0" stdDeviation="5" flood-color="#FF5700" flood-opacity="0.4"/>
    </filter>
  </defs>

  <rect width="100%" height="100%" fill="#1A1A1B"/>
  ${svgLines}
</svg>
`;

const assetsDir = path.join(__dirname, '..', 'assets');
if (!fs.existsSync(assetsDir)) fs.mkdirSync(assetsDir, { recursive: true });

const outputPath = path.join(assetsDir, 'redditminer_snoo.svg');
fs.writeFileSync(outputPath, svgContent, 'utf8');
console.log('Animated Reddit Snoo SVG generated at', outputPath);
