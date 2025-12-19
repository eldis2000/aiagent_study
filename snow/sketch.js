// 웹캠 관련 변수
let capture;
let camWidth = 640;
let camHeight = 480;

// 눈/꽃송이 객체 배열
let particles = []; // 변수명을 'particles'로 변경하여 유연하게 사용

// 배경음악 관련 변수
let bgm;

// ML5.js Handpose 모델 관련 변수
let handpose;
let predictions = []; // 손 인식 결과 저장 배열
let handDetected = false; // 손 감지 여부 플래그

// 입자의 현재 모양 (true: 꽃, false: 눈)
let isFlower = true; 

function preload() {
  // 사운드 파일 로드
  // ⚠️ 파일명을 확인하세요. (예: 'sound.mp3')
  bgm = loadSound("sound.mp3");
}

function setup() {
  let canvas = createCanvas(camWidth, camHeight);
  canvas.position(0, 40); 

  capture = createCapture(VIDEO);
  capture.size(camWidth, camHeight);
  capture.hide(); 
  
  // -------------------------------------
  // ML5.js Handpose 모델 로드
  // -------------------------------------
  handpose = ml5.handpose(capture, modelReady);
  // 손 감지 결과를 받을 콜백 함수 설정
  handpose.on('hand', gotResults); 

  // 초기 파티클 생성 (꽃 모양으로 시작)
  for (let i = 0; i < 150; i++) {
    particles.push(new Particle());
  }
  
  // 음악 재생 버튼
  const btnY = 50;
  
  const btn = createButton("▶ 재생");
  btn.position(10, btnY);
  btn.mousePressed(() => {
    if (!bgm.isPlaying()) {
      bgm.play();
    }
  });
  const pauseBtn = createButton("⏸ 일시정지");
  pauseBtn.position(70, btnY);
  pauseBtn.mousePressed(() => {
    if (bgm.isPlaying()) {
      bgm.pause();
    }
  });
  const stopBtn = createButton("■ 정지");
  stopBtn.position(156, btnY);
  stopBtn.mousePressed(() => {
    bgm.stop(); 
  });
}

// -------------------------------------
// ML5.js 콜백 함수
// -------------------------------------
function modelReady() {
  console.log('Handpose model loaded!');
}

function gotResults(results) {
  predictions = results;
  if (predictions.length > 0) {
    handDetected = true; // 손이 감지됨
  } else {
    handDetected = false; // 손이 감지되지 않음
  }
}

function draw() {
  if (capture.loadedmetadata) {
    // 캡처 영상을 반전시켜 거울처럼 보이게 할 수 있습니다. (선택 사항)
    // push();
    // translate(width, 0);
    // scale(-1, 1);
    image(capture, 0, 0, width, height);
    // pop();
  } else {
    background(0); 
  }

  // -------------------------------------
  // 손 감지 여부에 따라 꽃/눈 모드 전환
  // -------------------------------------
  if (handDetected && isFlower) { // 손이 감지되었고 현재 꽃 모드이면
    isFlower = false; // 눈 모드로 전환
    console.log("Hand detected! Switching to SNOW mode.");
    // 모든 파티클의 모드를 즉시 업데이트
    for (let p of particles) {
      p.setMode(false); // false: 눈
    }
  } else if (!handDetected && !isFlower) { // 손이 감지되지 않았고 현재 눈 모드이면
    isFlower = true; // 꽃 모드로 전환
    console.log("No hand detected. Switching to FLOWER mode.");
    // 모든 파티클의 모드를 즉시 업데이트
    for (let p of particles) {
      p.setMode(true); // true: 꽃
    }
  }

  // 모든 파티클 업데이트 및 그리기
  for (let p of particles) {
    p.update(); 
    p.display(); 
  }

  // (선택 사항) 손의 랜드마크를 화면에 그립니다.
  // drawKeypoints();
}

// (선택 사항) 손 랜드마크를 그리는 함수
function drawKeypoints() {
  for (let i = 0; i < predictions.length; i += 1) {
    const prediction = predictions[i];
    for (let j = 0; j < prediction.landmarks.length; j += 1) {
      const keypoint = prediction.landmarks[j];
      fill(0, 255, 0);
      noStroke();
      ellipse(keypoint[0], keypoint[1], 10, 10);
    }
  }
}


// 🌸/❄️ 파티클 클래스 (이름을 Particle로 변경)
class Particle {
  constructor() {
    this.posX = random(width);
    this.posY = random(height);
    
    // 기본 크기 설정
    this.baseSize = random(10, 20); // 꽃일 때의 크기
    this.snowSize = random(2, 5);   // 눈일 때의 크기

    this.currentSize = isFlower ? this.baseSize : this.snowSize; // 현재 크기
    
    // 크기에 따라 속도, 투명도 연관
    this.speed = map(this.currentSize, 2, 20, 0.5, 3.5); 
    this.opacity = map(this.currentSize, 2, 20, 150, 255); 
    
    // 꽃잎 색상
    this.petalColor = color(255, random(180, 230), random(180, 230), this.opacity); 
    
    // 회전 속도 및 초기 각도 (꽃일 때만 유효)
    this.rotationSpeed = random(-0.02, 0.02);
    this.currentRotation = random(TWO_PI); 

    // 마우스 X 위치에 따라 바람의 방향을 결정
    this.initialWind = random(-0.3, 0.3);

    this.isFlowerMode = isFlower; // 현재 모드 저장
  }

  // 모드 전환 함수
  setMode(mode) {
    this.isFlowerMode = mode;
    if (this.isFlowerMode) { // 꽃 모드
      this.currentSize = this.baseSize;
      this.speed = map(this.currentSize, 10, 20, 0.8, 3.5);
      this.opacity = map(this.currentSize, 10, 20, 150, 255);
      this.petalColor = color(255, random(180, 230), random(180, 230), this.opacity);
    } else { // 눈 모드
      this.currentSize = this.snowSize;
      this.speed = map(this.currentSize, 2, 5, 0.5, 2.5);
      this.opacity = map(this.currentSize, 2, 5, 150, 255);
      // 눈은 흰색
      this.petalColor = color(255, 255, 255, this.opacity); 
    }
  }

  update() {
    let windForce = map(mouseX, 0, width, -1, 1);
    
    this.posX += this.initialWind + windForce * 0.5; 
    this.posY += this.speed; 
    
    // 꽃 모드일 때만 회전
    if (this.isFlowerMode) {
      this.currentRotation += this.rotationSpeed; 
    }

    if (this.posY > height) {
      this.reset();
    }
    
    if (this.posX > width + this.currentSize / 2) {
      this.posX = -this.currentSize / 2;
    } else if (this.posX < -this.currentSize / 2) {
      this.posX = width + this.currentSize / 2;
    }
  }
  
  reset() {
      this.posY = -this.currentSize;
      this.posX = random(width);
      this.currentRotation = random(TWO_PI); // 재활용 시 회전 각도도 랜덤
      // 리셋 시에도 현재 모드를 유지하도록 합니다.
      this.setMode(this.isFlowerMode); 
  }

  display() {
    noStroke();
    fill(this.petalColor); 

    if (this.isFlowerMode) { // 🌸 꽃 그리기
      push(); 
      translate(this.posX, this.posY); 
      rotate(this.currentRotation); 

      let numPetals = 5; 
      let petalWidth = this.currentSize / 2.5; 
      let petalHeight = this.currentSize / 1.5; 

      for (let i = 0; i < numPetals; i++) {
        ellipse(0, this.currentSize / 4, petalWidth, petalHeight); 
        rotate(TWO_PI / numPetals); 
      }
      
      // 꽃술
      fill(255, 200, 0, this.opacity); 
      ellipse(0, 0, this.currentSize / 4); 

      pop(); 
    } else { // ❄️ 눈 그리기 (원형)
      ellipse(this.posX, this.posY, this.currentSize);
    }
  }
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
  // 캔버스 크기 변경 시 handpose 캡처 소스도 재설정 필요 (복잡해질 수 있어 여기서는 생략)
  // capture.size(width, height); 
}