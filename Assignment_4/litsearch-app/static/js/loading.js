const uploadForm = document.getElementById("upload-form")

uploadForm.addEventListener("submit", () => {

    document.querySelector(".dyn").style.display = "none"
    let loader = document.createElement("div")
    loader.innerHTML = "<div class='loading-circle'></div>"
    document.body.append(loader)

})  