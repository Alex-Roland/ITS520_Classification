async function runExample() {

  const x = new Float32Array(4);
  x[0] = parseFloat(document.getElementById('box0c1').value) || 0;
  x[1] = parseFloat(document.getElementById('box1c1').value) || 0;
  x[2] = parseFloat(document.getElementById('box2c1').value) || 0;
  x[3] = parseFloat(document.getElementById('box3c1').value) || 0;

  const tensorX = new ort.Tensor('float32', x, [1, 4]);

  try {
    const session = await ort.InferenceSession.create("temperature_humidity_data.onnx?v=" + Date.now());
    const results = await session.run({ input1: tensorX });
    const output = results.output1.data;
    let season = "";

    if (output[0].toFixed(2) == 0.00) {
      season = "Winter";
    } else if (output[0].toFixed(2) == 0.01) {
      season = "Spring";
    } else if (output[0].toFixed(2) == 0.02) {
      season = "Summer";
    } else if (output[0].toFixed(2) == 0.03) {
      season = "Fall";
    }

    // render here (output is in scope)
    const predictions = document.getElementById('predictions');
    predictions.innerHTML = `
      <table>
        <tr><td> Season </td>          <td id="c1td0"> <b>${season}</b> </td></tr>
      </table>`;
  } catch (e) {
    console.error("ONNX runtime error:", e);
    alert("Error: " + e.message);
  }
}
