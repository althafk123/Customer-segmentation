
const demoButton = document.getElementById("demo-button");
const incomeInput = document.getElementById("income");
const spendingInput = document.getElementById("spending");
const customerForm = document.getElementById("customer-form");

demoButton.addEventListener("click", function () {
    incomeInput.value = 50;
    spendingInput.value = 75;

    document.getElementById("predict").scrollIntoView({
        behavior: "smooth",
        block: "start"
    });

    spendingInput.focus();
});

customerForm.addEventListener("submit", function (event) {
    const income = Number(incomeInput.value);
    const spending = Number(spendingInput.value);

    if (!Number.isFinite(income) || income < 0 || income > 1000) {
        event.preventDefault();
        alert("Enter an income between 0 and 1000.");
        return;
    }

    if (!Number.isFinite(spending) || spending < 1 || spending > 100) {
        event.preventDefault();
        alert("Enter a spending score between 1 and 100.");
    }
});
