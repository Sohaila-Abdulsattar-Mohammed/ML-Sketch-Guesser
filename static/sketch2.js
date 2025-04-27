//this file handles the frontend of the drawing canvas in a way that is compatible with the RNN model in the backend
//it captures the user's drawing as a sequence of strokes, processes it, and sends it to the backend for prediction

//list of class names representing possible drawing categories
let classNames = [
  'cat', 'tree', 'fish', 'clock', 'castle', 'crown', 'lollipop', 'moon',
  'watermelon', 'tornado', 'apple', 'bowtie', 'bicycle', 'diamond',
  'flower', 'butterfly', 'eye', 'lightning', 'cloud', 'pizza'
];

//array to store confidence values for each class, initialized to 0
let barData = new Array(classNames.length).fill(0);

//retrieve the target word (prompted word) from localStorage
let targetWord = localStorage.getItem('promptWord');

//flag to track if the user has won
let hasWon = false;

//display the target word in uppercase on the page
document.getElementById("target-word").textContent = targetWord.toUpperCase();

//initialize the bar chart using ApexCharts
let chart = new ApexCharts(document.querySelector("#chart"), {
  chart: {
    type: 'bar', //bar chart type
    height: '90%', //chart height
    animations: { //smooth animations for updates
      enabled: true,
      easing: 'easeinout',
      dynamicAnimation: {
        speed: 300 //animation speed
      }
    },
    toolbar: { show: false } //disable toolbar for simplicity
  },
  plotOptions: {
    bar: {
      horizontal: true, //horizontal bar chart
      barHeight: '90%' //bar height
    }
  },
  dataLabels: { enabled: false }, //disable data labels for a cleaner look
  xaxis: {
    categories: classNames, //use class names as x-axis labels
    min: 0, //minimum value for x-axis
    max: 1, //maximum value for x-axis (confidence range is 0-1)
    tickAmount: 5, //number of ticks on the x-axis
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
        fontWeight: 'bold', //bold font for better visibility
        fontSize: '14px' //font size
      }
    }
  },
  series: [{ data: barData }], //initial data for the chart
  colors: ['#f9a9ee'] //bar color
});
chart.render(); //render the chart

//timer logic
let timeLeft = 20; //20 seconds countdown
let timer = setInterval(() => {
  timeLeft--;
  document.getElementById("timer").textContent = `00:${timeLeft < 10 ? '0' : ''}${timeLeft}`; //update timer display
  if (timeLeft <= 0) { //if time runs out
    clearInterval(timer); //stop the timer
    if (!hasWon) { //if the user hasn't won
      const audio = new Audio("/static/lose.wav"); //play lose sound
      audio.play().then(() => {
        setTimeout(() => {
          window.location.href = "lose"; //redirect to lose page
        }, audio.duration * 1000); //wait for sound to finish
      }).catch(err => {
        console.warn("Autoplay blocked, skipping sound:", err);
        window.location.href = "lose"; //redirect immediately if sound fails
      });
    }
  }
}, 1000); //update every second

//variables for prediction timing
let lastPredictionTime = 0; //last time a prediction was made
let predictionDelay = 1500; //minimum delay between predictions (1.5 seconds)

//variables for tracking drawing bounds
let minX, minY, maxX, maxY;

//variables for strokes
let strokes = []; //array of completed strokes
let currentStroke = null; //current stroke being drawn
let prevX = 0; //previous x-coordinate
let prevY = 0; //previous y-coordinate

//p5.js setup function to initialize the canvas
function setup() {
  pixelDensity(1); //set pixel density for consistent rendering
  let c = createCanvas(window.innerWidth / 2.5, window.innerWidth / 2.5); //create a square canvas
  c.parent(document.querySelector('.left')); //attach canvas to the left container
  background(255); //set background to white
  strokeWeight(20); //set stroke weight for drawing
  stroke(0); //set stroke color to black
  //initialize bounds of the drawing
  minX = width; //initialize minX to canvas width
  minY = height; //initialize minY to canvas height
  maxX = 0; //initialize maxX to 0
  maxY = 0; //initialize maxY to 0
}

//p5.js draw function to handle drawing on the canvas
function draw() {
  if (mouseIsPressed && mouseX >= 0 && mouseX <= width && mouseY >= 0 && mouseY <= height) {
    line(pmouseX, pmouseY, mouseX, mouseY); //draw a line between previous and current mouse positions
    minX = Math.min(minX, mouseX); //update minX
    maxX = Math.max(maxX, mouseX); //update maxX
    minY = Math.min(minY, mouseY); //update minY
    maxY = Math.max(maxY, mouseY); //update maxY
  }
}

//p5.js mousePressed function to start a new stroke
function mousePressed() {
  if (mouseX >= 0 && mouseX <= width && mouseY >= 0 && mouseY <= height) {
    currentStroke = { x: [], y: [] }; //initialize a new stroke
    currentStroke.x.push(mouseX); //add the starting x-coordinate
    currentStroke.y.push(mouseY); //add the starting y-coordinate
  }
}

//p5.js mouseDragged function to continue the current stroke
function mouseDragged() {
  if (currentStroke && mouseX >= 0 && mouseX <= width && mouseY >= 0 && mouseY <= height) {
    currentStroke.x.push(mouseX); //add the current x-coordinate
    currentStroke.y.push(mouseY); //add the current y-coordinate
  }

  let now = millis(); //get the current time
  if (now - lastPredictionTime > predictionDelay) { //check if enough time has passed
    sendToModel(); //send the drawing to the model for prediction
    lastPredictionTime = now; //update the last prediction time
  }
}

//p5.js mouseReleased function to finish the current stroke
function mouseReleased() {
  if (currentStroke) {
    strokes.push(currentStroke); //add the current stroke to the list of strokes
    currentStroke = null; //reset the current stroke
  }
  sendToModel(); //send the drawing to the model for prediction
}

//the following two functions help process the drawing/strokes in such a way that it matches the input the model expects (so that the input is formatted like the SketchRNN data)
//this is done in accordance with how the data was preprocessed as explained in the SketchRNN paper

//helper function to simplify a stroke using the Ramer-Douglas-Peucker algorithm, which is a common method to simplify lines and curves by removing "unnecessary" points that don't add much to the overall shape, again this is done as per the preprocessing of the SketchRNN data
function simplifyStroke(xs, ys, epsilon = 2) {
  const points = xs.map((x, i) => ({ x, y: ys[i] })); //convert to point objects
  const simp = simplify(points, epsilon, true); //simplify the stroke, the simplify function is from the simplify-js library
  return {
    x: simp.map(p => p.x), //extract simplified x-coordinates
    y: simp.map(p => p.y) //extract simplified y-coordinates
  };
}

//helper function to resample a stroke so that points are at least 1px apart, again this is done as per the preprocessing of the SketchRNN data
function resampleStroke(xs, ys) {
  const outX = [xs[0]], outY = [ys[0]]; //start with the first point
  for (let i = 1; i < xs.length; ++i) {
    const dx = xs[i] - outX[outX.length - 1]; //calculate x-distance
    const dy = ys[i] - outY[outY.length - 1]; //calculate y-distance
    const dist = Math.hypot(dx, dy); //calculate Euclidean distance
    if (dist >= 1.0) { //if the distance is at least 1px
      outX.push(xs[i]); //add the x-coordinate
      outY.push(ys[i]); //add the y-coordinate
    }
  }
  return { x: outX, y: outY }; //return the resampled stroke
}

//function to send the drawing to the backend model for prediction
function sendToModel() {
  if (strokes.length === 0 && !currentStroke) return; //if no strokes, do nothing

  //gather all points from all strokes
  const allX = [], allY = [];
  const allStrokes = strokes.concat(currentStroke ? [currentStroke] : []); //if currentStroke exists (user is currently drawing), we wrap it in an array [currentStroke] and add it to strokes; otherwise we add an empty array
  //loop through every stroke object
  for (const s of allStrokes) {
    allX.push(...s.x); allY.push(...s.y); //now allX and allY are flat arrays that contain all the x and y points from all strokes, combined. ... is a spread syntax btw
  }

  //getting the bounding box of the drawing
  //using the spread syntax to pass the entire array as individual arguments
  const minX = Math.min(...allX), maxX = Math.max(...allX);
  const minY = Math.min(...allY), maxY = Math.max(...allY);

  //calculate how much we need to scale the drawing to fit within a 255x255 box, while keeping the aspect ratio intact; recall that in the QuickDraw dataset that SketchRNN used, they uniformly scaled the drawing to have a maximum value of 255
  const scale = 255 / Math.max(maxX - minX, maxY - minY || 1);
  //find the center point of the drawing so we can shift everything to be centered around (0, 0) before scaling
  const cx = (maxX + minX) / 2, cy = (maxY + minY) / 2;

  //build a sequence of (dx, dy, pen) for the model
  const seq = [];
  //keep track of the last position, so we can calculate delta movements
  let prevX = 0, prevY = 0;

  //a helper to add a step to the sequence
  const push = (dx, dy, pen) => {
    seq.push([dx, dy, pen]); //add the step to the sequence
    prevX += dx; prevY += dy; //update the previous position
  };

  for (const raw of allStrokes) { //recall, allStrokes is an array of stroke objects, each "raw" stroke has two arrays: raw.x and raw.y, the list of x and y coordinates the user drew
    let s = simplifyStroke(raw.x, raw.y, 2.0); //simplify the stroke
    s = resampleStroke(s.x, s.y); //takes the simplified stroke and resample it

    //loop through each point in the cleaned stroke
    for (let i = 0; i < s.x.length; ++i) {
      const nx = (s.x[i] - cx) * scale; //normalize x-coordinate
      const ny = (s.y[i] - cy) * scale; //normalize y-coordinate
      let dx = Math.round(nx - prevX); //calculate delta x
      let dy = Math.round(ny - prevY); //calculate delta y
      dx = Math.max(-255, Math.min(255, dx)); //clamp dx to [-255, 255]
      dy = Math.max(-255, Math.min(255, dy)); //clamp dy to [-255, 255]
      push(dx, dy, 0); //add pen-down stroke
    }
    push(0, 0, 1); //add pen-up stroke
  }

  // console.log("normalized seq", seq); //log the normalized sequence

  const chosenModel = localStorage.getItem("chosenModel"); //get the chosen model

  //getting the current origin of the page
  const CURRENT_ORIGIN = window.location.origin;

  const API_BASE = CURRENT_ORIGIN.includes("localhost") ? "http://localhost:5000" : CURRENT_ORIGIN;

  //send the sequence to the backend for prediction
  fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sequence: seq, model: chosenModel })
  })
  .then(r => r.json())
  .then((data) => {
    // console.log("Predictions:", data); //log the predictions

    //update the bar chart with new confidence values
    let newData = new Array(classNames.length).fill(0);
    data.forEach(entry => {
      let idx = classNames.indexOf(entry.label.toLowerCase()); //find index of predicted label
      if (idx !== -1) newData[idx] = entry.confidence; //update confidence value
    });
    chart.updateSeries([{ data: newData }]); //update chart

    //check if the top prediction matches the target word
    const topGuess = data[0]; //top prediction
    const guessedCorrectly = topGuess.label.toLowerCase() === targetWord.toLowerCase(); //check match
    const confidence = topGuess.confidence; //confidence of top prediction

    if (!hasWon && guessedCorrectly && confidence >= 0.8) { //if correct and confident enough
      hasWon = true; //mark as won
      clearInterval(timer); //stop the timer

      const audio = new Audio("/static/win.wav"); //play win sound
      audio.play().then(() => {
        setTimeout(() => {
          window.location.href = "win"; //redirect to win page
        }, audio.duration * 1000); //wait for sound to finish
      }).catch(err => {
        console.warn("Autoplay blocked, skipping sound:", err);
        window.location.href = "win"; //redirect immediately if sound fails
      });
    }
  })
  .catch(console.error); //log any errors
}
