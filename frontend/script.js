let file;

document.getElementById("fileInput").addEventListener("change", function (e) {
  file = e.target.files[0];

  const preview = document.getElementById("preview");
  preview.src = URL.createObjectURL(file);
  preview.style.display = "block";
});

async function detect() {
  if (!file) {
    alert("Image upload karo!");
    return;
  }

  const formData = new FormData();
  formData.append("image", file);

  document.getElementById("status").innerText = "⏳ Detecting...";

  const res = await fetch("http://localhost:5000/predict", {
    method: "POST",
    body: formData
  });

  const data = await res.json();

  document.getElementById("resultImg").src =
    "data:image/jpeg;base64," + data.result;

  document.getElementById("status").innerText =
    data.status === "OK"
      ? "✅ Product OK"
      : "❌ Defect Detected";
}