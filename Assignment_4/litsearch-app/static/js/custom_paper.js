fetch("/encode-text", {
    method: "GET"
})

const tags = document.querySelectorAll(".tags")

window.addEventListener("click", (e) => {
    if (e.target.classList.contains("tag")) {
        searchBar.value = e.target.innerText
        make_a_search()
        resultsContainer.style.display = "block"
    }
})

const subjectSearcher = document.getElementById("subject-finder")
const subjectResult = document.getElementById("subject-result")

subjectSearcher.addEventListener("keyup", () => {

    document.getElementById("loading-div").style.display = "block"

    fetch("/subject-search", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            "subject": subjectSearcher.value
        })
    })
    .then(response => {
        response.json().then(data => {
            document.getElementById("loading-div").innerHTML = ""
            subjectResult.innerText = data.text
        })
    })

})