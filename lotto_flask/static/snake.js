const canvas = document.getElementById("snake");
const ctx = canvas.getContext("2d");

const box = 15;
let snake = [];
snake[0] = { x: 10 * box, y: 10 * box };
let food = {
  x: Math.floor(Math.random() * 20) * box,
  y: Math.floor(Math.random() * 20) * box,
};
let direction;
let score = 0;

document.addEventListener("keydown", directionHandler);

function directionHandler(event) {
  if (event.key === "ArrowLeft" && direction !== "RIGHT") direction = "LEFT";
  else if (event.key === "ArrowUp" && direction !== "DOWN") direction = "UP";
  else if (event.key === "ArrowRight" && direction !== "LEFT") direction = "RIGHT";
  else if (event.key === "ArrowDown" && direction !== "UP") direction = "DOWN";
}

function draw() {
  ctx.fillStyle = "#0f172a";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  for (let i = 0; i < snake.length; i++) {
    ctx.fillStyle = i === 0 ? "#3b82f6" : "#60a5fa";
    ctx.fillRect(snake[i].x, snake[i].y, box, box);
  }

  ctx.fillStyle = "#facc15";
  ctx.fillRect(food.x, food.y, box, box);

  let snakeX = snake[0].x;
  let snakeY = snake[0].y;

  if (direction === "LEFT") snakeX -= box;
  if (direction === "UP") snakeY -= box;
  if (direction === "RIGHT") snakeX += box;
  if (direction === "DOWN") snakeY += box;

  if (snakeX === food.x && snakeY === food.y) {
    score++;
    food = {
      x: Math.floor(Math.random() * 20) * box,
      y: Math.floor(Math.random() * 20) * box,
    };
  } else {
    snake.pop();
  }

  const newHead = { x: snakeX, y: snakeY };

  if (
    snakeX < 0 ||
    snakeY < 0 ||
    snakeX >= canvas.width ||
    snakeY >= canvas.height ||
    collision(newHead, snake)
  ) {
    clearInterval(game);
    ctx.fillStyle = "#f87171";
    ctx.font = "20px Inter";
    ctx.fillText("💀 게임 오버!", 80, 150);
    return;
  }

  snake.unshift(newHead);

  ctx.fillStyle = "#cbd5e1";
  ctx.font = "16px Inter";
  ctx.fillText("점수: " + score, 10, canvas.height - 10);
}

function collision(head, array) {
  return array.some(segment => segment.x === head.x && segment.y === head.y);
}

const game = setInterval(draw, 120);
