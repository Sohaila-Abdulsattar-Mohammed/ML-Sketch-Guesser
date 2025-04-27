//this file handles the frontend of the drawing canvas in a way that is compatible with the CNN model in the backend
//it captures the user's drawing as an image, processes it, and sends it to the backend for prediction

//list of class names representing possible categories for predictions
let classNames = [
  'cat', 'tree', 'fish', 'clock', 'castle', 'crown', 'lollipop', 'moon',
  'watermelon', 'tornado', 'apple', 'bowtie', 'bicycle', 'diamond',
  'flower', 'butterfly', 'eye', 'lightning', 'cloud', 'pizza'
];

//initialize bar chart data with zeros for each class
let barData = new Array(classNames.length).fill(0);

//retrieve the target word (prompted word) from local storage
let targetWord = localStorage.getItem('promptWord');

//boolean to track if the user has won
let hasWon = false;

//display the target word in uppercase on the page
document.getElementById("target-word").textContent = targetWord.toUpperCase();

//initialize the bar chart using ApexCharts
let chart = new ApexCharts(document.querySelector("#chart"), {
  chart: {
    type: 'bar', //bar chart type
    height: '90%', //chart height
    animations: { //enable smooth animations
      enabled: true,
      easing: 'easeinout',
      dynamicAnimation: {
        speed: 300 //animation speed
      }
    },
    toolbar: { show: false } //disable toolbar
  },
  plotOptions: {
    bar: {
      horizontal: true, //horizontal bar chart
      barHeight: '90%' //bar height
    }
  },
  dataLabels: { enabled: false }, //disable data labels
  xaxis: {
    categories: classNames, //use class names as x-axis categories
    min: 0, //minimum value for x-axis
    max: 1, //maximum value for x-axis
    tickAmount: 5, //number of ticks on x-axis
    labels: { 
        show: true,
        formatter: val => `${(val * 100).toFixed(0)}%` //format as percentage
      }
  },
  yaxis: {
    labels: {
      style: {
        colors: '#000', //default label color
        colors: classNames.map(name => 
          name === targetWord.toLowerCase() ? '#e102a6' : '#000' //highlight target word
        ),
        fontWeight: 'bold', //bold font
        fontSize: '14px' //font size
      }
    }
  },
  series: [{ data: barData }], //initial bar data
  colors: ['#f9a9ee'] //bar color
});
chart.render(); //render the chart

//timer setup
let timeLeft = 20; //countdown timer starts at 20 seconds
let timer = setInterval(() => {
  timeLeft--;
  document.getElementById("timer").textContent = `00:${timeLeft < 10 ? '0' : ''}${timeLeft}`; //update timer display
  if (timeLeft <= 0) { //if time runs out
    clearInterval(timer); //stop the timer
    if (!hasWon) { //if the user hasn't won
      const audio = new Audio("lose.wav"); //play lose sound
      audio.play().then(() => {
          setTimeout(() => {
          window.location.href = "lose.html"; //redirect to lose page
          }, audio.duration * 1000); //wait for sound to finish
      }).catch(err => {
          console.warn("Autoplay blocked, skipping sound:", err);
          window.location.href = "lose.html"; //redirect if autoplay fails
      });
    }
  }
}, 1000); //update every second

//variables for prediction timing and drawing bounds
let lastPredictionTime = 0; //last time a prediction was made
let predictionDelay = 1500; //minimum delay between predictions (1.5 seconds)
let minX, minY, maxX, maxY; //bounds of the drawing

//p5.js setup function to initialize the canvas
function setup() {
  pixelDensity(1); //set pixel density to 1 for consistent rendering
  let c = createCanvas(window.innerWidth / 2.5, window.innerWidth / 2.5); //create a square canvas
  c.parent(document.querySelector('.left')); //attach canvas to the left container
  background(255); //set background to white
  strokeWeight(20); //set stroke weight for drawing
  stroke(0); //set stroke color to black
  //initialize bounds of the drawing
  minX = width;
  minY = height;
  maxX = 0;
  maxY = 0;
}

//p5.js draw function to handle drawing on the canvas
function draw() {
  //if the mouse is pressed and within canvas bounds
  if (mouseIsPressed && mouseX >= 0 && mouseX <= width && mouseY >= 0 && mouseY <= height) {
    //draw a line from the previous mouse position to the current mouse position, ie draw a line following the mouse
    line(pmouseX, pmouseY, mouseX, mouseY);
    //update bounds of the drawing
    minX = Math.min(minX, mouseX);
    maxX = Math.max(maxX, mouseX);
    minY = Math.min(minY, mouseY);
    maxY = Math.max(maxY, mouseY);
  }
}

//handle mouse dragging to trigger predictions
function mouseDragged() {
  let now = millis(); //current time in milliseconds
  if (now - lastPredictionTime > predictionDelay) { //check if enough time has passed
    sendToModel(); //send drawing to model for prediction
    lastPredictionTime = now; //update last prediction time
  }
}

//handle mouse release to trigger a prediction
function mouseReleased() {
  sendToModel(); //send drawing to model for prediction
}

//function to send the drawing to the model for prediction
function sendToModel() {

  //expand the bounding box of the drawing with padding
  const pad = 10; //padding size
  //ensuring bounds stay within canvas
  minX = Math.max(0, minX - pad);
  minY = Math.max(0, minY - pad);
  maxX = Math.min(width, maxX + pad);
  maxY = Math.min(height, maxY + pad);

  let boxW = maxX - minX; //width of the bounding box
  let boxH = maxY - minY; //height of the bounding box
  const size = Math.max(boxW, boxH); //ensuring square crop

  const offsetX = Math.max(0, Math.floor((boxW < size) ? minX - (size - boxW) / 2 : minX)); //center crop horizontally
  const offsetY = Math.max(0, Math.floor((boxH < size) ? minY - (size - boxH) / 2 : minY)); //center crop vertically

  const squareCrop = get(offsetX, offsetY, size, size); //extract the cropped region

  //gradually downsample the cropped image to 28x28, doing so gradually helps preserve visual quality
  const step1 = createGraphics(104, 104); //intermediate size
  step1.pixelDensity(1);
  step1.noSmooth(); //disable smoothing for sharp edges
  step1.background(255); //white background
  step1.image(squareCrop, 0, 0, 104, 104); //resize to 104x104

  const step2 = createGraphics(52, 52); //smaller size
  step2.pixelDensity(1);
  step2.noSmooth();
  step2.background(255);
  step2.image(step1, 0, 0, 52, 52); //resize to 52x52

  const final = createGraphics(28, 28); //final size
  final.pixelDensity(1);
  final.noSmooth();
  final.background(255);
  final.image(step2, 1, 1, 26, 26); //resize to 28x28 with slight inset

  final.loadPixels(); //load pixel data for processing

  //convert the image to grayscale, normalize, and invert
  const input = [];
  for (let i = 0; i < final.pixels.length; i += 4) {
    const gray = 255 - final.pixels[i]; //invert white background
    input.push(gray / 255); //normalize to [0, 1]
  }

  //you can uncomment the following lines to visualize the processed (downsized) image on the page; it is not well formatted on the page but this is useful for debugging and understanding what gets sent to the model
  
  // const ctx = document.getElementById("preview").getContext("2d");
  // const imgData = ctx.createImageData(28, 28);
  // for (let i = 0; i < input.length; i++) {
  //   const val = 255 - input[i] * 255;
  //   imgData.data[i * 4 + 0] = val;
  //   imgData.data[i * 4 + 1] = val;
  //   imgData.data[i * 4 + 2] = val;
  //   imgData.data[i * 4 + 3] = 255;
  // }
  // ctx.putImageData(imgData, 0, 0);
  // ctx.imageSmoothingEnabled = false;
  // ctx.drawImage(ctx.canvas, 0, 0, 28, 28, 0, 0, 56, 56);

  //send the processed image to the backend for prediction
  const chosenModel = localStorage.getItem("chosenModel"); //retrieve chosen model from local storage
  fetch("http://localhost:5000/predict", { //send POST request to backend
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ 
      pixels: input, //send pixel data
      model: chosenModel //include chosen model
     })
  })
    .then(res => res.json()) //parse JSON response
    .then((data) => {
      console.log("Predictions:", data); //log predictions

      //update the bar chart with new prediction data
      let newData = new Array(classNames.length).fill(0); //reset data
      data.forEach(entry => {
        let idx = classNames.indexOf(entry.label.toLowerCase()); //find index of predicted label
        if (idx !== -1) newData[idx] = entry.confidence; //update confidence value
      });
      chart.updateSeries([{ data: newData }]); //update chart

      //check if the top prediction matches the target word
      const topGuess = data[0]; //top prediction
      const guessedCorrectly = topGuess.label.toLowerCase() === targetWord.toLowerCase(); //check match
      const confidence = topGuess.confidence; //confidence of top prediction

      //if the prediction is correct and confident enough (over 80%)
      if (!hasWon && guessedCorrectly && confidence >= 0.8) {
        hasWon = true; //mark as won
        clearInterval(timer); //stop the timer

        const audio = new Audio("win.wav"); //play win sound
        audio.play().then(() => {
          setTimeout(() => {
          window.location.href = "win.html"; //redirect to win page
          }, audio.duration * 1000); //wait for sound to finish
        }).catch(err => {
          console.warn("Autoplay blocked, skipping sound:", err);
          window.location.href = "win.html"; //redirect if autoplay fails
        });
      }
    })
    .catch(console.error); //log errors
}
