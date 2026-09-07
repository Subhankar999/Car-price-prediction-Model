const API_URL = "https://car-price-prediction-model-hgja.onrender.com/predict";

// ABtest is required by the trained model but is not shown to the user.
const FIXED_AB_TEST_VALUE = "test";

const GAUGE_MAX = 1500000;

const form = document.getElementById("predictionForm");
const loading = document.getElementById("loading");
const error = document.getElementById("error");
const submitBtn = document.getElementById("submitBtn");
const needle = document.getElementById("needle");
const gaugeFill = document.getElementById("gaugeFill");
const readoutValue = document.getElementById("readoutValue");
const readoutLabel = document.getElementById("readoutLabel");


function setGauge(prediction) {
    const pct = Math.max(0, Math.min(1, prediction / GAUGE_MAX));

    const angle = -90 + pct * 180;

    needle.style.transform = `rotate(${angle}deg)`;

    const dash = 314;

    gaugeFill.style.strokeDashoffset = String(
        dash * (1 - pct)
    );

    gaugeFill.classList.add("is-filled");
}


function resetGauge() {
    needle.style.transform = "rotate(-90deg)";

    gaugeFill.style.strokeDashoffset = "314";

    gaugeFill.classList.remove("is-filled");

    readoutValue.classList.remove("is-set");

    readoutValue.textContent = "₹ —";

    readoutLabel.textContent = "Estimated value";
}


form.addEventListener("submit", async function (event) {

    event.preventDefault();

    error.style.display = "none";

    loading.textContent = "Getting your appraisal…";

    submitBtn.disabled = true;


    // Render may take some time to wake up if the server was idle.
    const wakeupTimer = setTimeout(() => {

        loading.textContent =
            "Still working… the server may be waking up from sleep (can take up to a minute).";

    }, 6000);


    const data = {

        seller: document.getElementById("seller").value,

        ABtest: FIXED_AB_TEST_VALUE,

        vehicle_type:
            document.getElementById("vehicle_type").value,

        year_of_registration:
            Number(
                document.getElementById("year_of_registration").value
            ),

        powerPS:
            Number(
                document.getElementById("power_ps").value
            ),

        kilometer:
            Number(
                document.getElementById("kilometer").value
            ),

        gearbox:
            document.getElementById("gearbox").value,

        fuelType:
            document.getElementById("fuel_type").value,

        brand:
            document.getElementById("brand").value,

        model:
            document.getElementById("model").value,

        notRepairedDamage:
            document.getElementById("not_repaired_damage").value
    };


    try {

        const response = await fetch(API_URL, {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify(data)
        });


        const responseData = await response.json();


        if (!response.ok) {

            throw new Error(
                responseData.detail || "Prediction failed"
            );

        }


        clearTimeout(wakeupTimer);


        const prediction =
            Number(responseData.predicted_price);


        setGauge(prediction);


        readoutLabel.textContent =
            "Estimated value";


        readoutValue.textContent =
            "₹ " + prediction.toLocaleString("en-IN");


        readoutValue.classList.add("is-set");


        loading.textContent = "";


    } catch (err) {

        clearTimeout(wakeupTimer);

        loading.textContent = "";

        error.textContent =
            "Couldn't get an appraisal: " + err.message;

        error.style.display = "block";


    } finally {

        submitBtn.disabled = false;

    }

});


resetGauge();
