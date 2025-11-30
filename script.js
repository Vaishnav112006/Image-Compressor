document.addEventListener("DOMContentLoaded", function(){
  const themeToggle = document.getElementById("themeToggle");
  const body = document.body;
  if (localStorage.getItem("theme") === "dark") {
    body.classList.remove("light"); body.classList.add("dark");
    themeToggle.checked = true;
  } else { body.classList.add("light"); }
  themeToggle?.addEventListener("change", function(){
    if (this.checked) { body.classList.remove("light"); body.classList.add("dark"); localStorage.setItem("theme","dark"); }
    else { body.classList.remove("dark"); body.classList.add("light"); localStorage.setItem("theme","light"); }
  });

  const qrange = document.getElementById("qrange");
  const qval = document.getElementById("qval");
  if(qrange && qval){
    qval.innerText = qrange.value;
    qrange.addEventListener("input", function(){ qval.innerText = this.value; });
  }
});