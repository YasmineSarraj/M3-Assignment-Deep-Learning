const searchBar = document.getElementById("search-bar")
const resultsContainer = document.querySelector(".results")

searchBar.addEventListener("focus", () => {
    resultsContainer.style.display = "block"
})

function make_a_search() {

    fetch("/search", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            "current_search": searchBar.value
        })
    })
    .then(response => {response.json()
        .then(data => {
            resultsContainer.innerHTML = ""
            data.forEach(element => {
                let resultItem = document.createElement("div")
                resultItem.classList.add("result")
                if (element.type == "topic") {
                resultItem.innerHTML = `
<a href="#">
<svg width="11" height="16.5" viewBox="0 0 20 29" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M15.9773 14.0979C16.7955 12.9311 17.2727 11.5094 17.2727 9.96875C17.2727 5.96426 14.017 2.71875 10 2.71875C5.98295 2.71875 2.72727 5.96426 2.72727 9.96875C2.72727 11.5094 3.20455 12.9311 4.02273 14.0979C4.23295 14.398 4.48295 14.7379 4.75 15.1004C5.48295 16.1029 6.35795 17.3037 7.01136 18.4875C7.60227 19.5637 7.90341 20.6852 8.05114 21.7443H5.28409C5.15909 21.0646 4.94886 20.402 4.61364 19.7902C4.05114 18.7707 3.35227 17.8135 2.65341 16.8563C2.35795 16.4541 2.0625 16.052 1.77841 15.6441C0.659091 14.0412 0 12.0814 0 9.96875C0 4.46328 4.47727 0 10 0C15.5227 0 20 4.46328 20 9.96875C20 12.0814 19.3409 14.0412 18.2159 15.6498C17.9318 16.0576 17.6364 16.4598 17.3409 16.8619C16.642 17.8135 15.9432 18.7707 15.3807 19.7959C15.0455 20.4076 14.8352 21.0703 14.7102 21.75H11.9545C12.1023 20.6908 12.4034 19.5637 12.9943 18.4932C13.6477 17.3094 14.5227 16.1086 15.2557 15.1061C15.5227 14.7436 15.767 14.4037 15.9773 14.1035V14.0979ZM10 7.25C8.49432 7.25 7.27273 8.46777 7.27273 9.96875C7.27273 10.4672 6.86364 10.875 6.36364 10.875C5.86364 10.875 5.45455 10.4672 5.45455 9.96875C5.45455 7.46523 7.48864 5.4375 10 5.4375C10.5 5.4375 10.9091 5.84531 10.9091 6.34375C10.9091 6.84219 10.5 7.25 10 7.25ZM10 29C7.48864 29 5.45455 26.9723 5.45455 24.4688V23.5625H14.5455V24.4688C14.5455 26.9723 12.5114 29 10 29Z" fill="#3D6C8D"/>
</svg>
<span>${element.item}</span>
</a>
                `}
                else if (element.type == "paper") {
                    resultItem.innerHTML = `
<a href="#">
<svg width="11" height="16.5" viewBox="0 0 20 28" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M3.33333 24.5221C2.875 24.5221 2.5 24.1415 2.5 23.6765V3.38235C2.5 2.91728 2.875 2.53676 3.33333 2.53676H11.6667V6.76471C11.6667 7.70014 12.4115 8.45588 13.3333 8.45588H17.5V23.6765C17.5 24.1415 17.125 24.5221 16.6667 24.5221H3.33333ZM3.33333 0C1.49479 0 0 1.51677 0 3.38235V23.6765C0 25.542 1.49479 27.0588 3.33333 27.0588H16.6667C18.5052 27.0588 20 25.542 20 23.6765V8.16521C20 7.26677 19.651 6.40533 19.026 5.77114L14.3073 0.988281C13.6823 0.35409 12.8385 0 11.9531 0H3.33333ZM6.25 13.5294C5.55729 13.5294 5 14.0949 5 14.7978C5 15.5007 5.55729 16.0662 6.25 16.0662H13.75C14.4427 16.0662 15 15.5007 15 14.7978C15 14.0949 14.4427 13.5294 13.75 13.5294H6.25ZM6.25 18.6029C5.55729 18.6029 5 19.1684 5 19.8713C5 20.5742 5.55729 21.1397 6.25 21.1397H13.75C14.4427 21.1397 15 20.5742 15 19.8713C15 19.1684 14.4427 18.6029 13.75 18.6029H6.25Z" fill="#3D6C8D"/>
</svg>
<span>${element.item}</span>
</a>
                    `
                }
                resultsContainer.append(resultItem)
            });
        })
    })

}

searchBar.addEventListener("keyup", () => {
    make_a_search()
})


window.addEventListener("click", (e) => {

    if (e.target.classList.contains("result") || e.target.id != "search-bar") {
        resultsContainer.style.display = "none"
    }

})