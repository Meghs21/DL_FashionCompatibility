async function checkCompatibility() {

  const top = document.getElementById("top").files[0];
  const bottom = document.getElementById("bottom").files[0];
  const footwear = document.getElementById("footwear").files[0];
  const accessories = document.getElementById("accessories").files[0];

  if (!top || !bottom) {
    alert("Top and Bottom required!");
    return;
  }

  const formData = new FormData();
  formData.append("top", top);
  formData.append("bottom", bottom);

  if (footwear) formData.append("footwear", footwear);
  if (accessories) formData.append("accessories", accessories);

  try {
    document.getElementById("score").innerText = "Loading...";

    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      body: formData
    });

    const data = await response.json();

    console.log(data); // DEBUG

    const score = data.compatibility || 0;

    // Update text
    document.getElementById("score").innerText = score + "%";

    // Update circle
    document.getElementById("circle").style.background =
      `conic-gradient(#22c55e ${score}%, #374151 ${score}%)`;

    // Feedback
    document.getElementById("feedback").innerText =
      data.feedback || "No feedback";

  } catch (err) {
    alert("Backend error");
    console.error(err);
  }
}