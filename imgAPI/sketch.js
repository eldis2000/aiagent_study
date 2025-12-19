// Flask 서버가 실행 중인 로컬 주소(5000번 포트)를 지정합니다.
let serverUrl = 'http://127.0.0.1:5000/chat'; 

let fileInput;     // 로컬 파일 선택 필드
let promptField;   // 질문(프롬프트) 입력 필드
let sendButton;
let responseText;

function setup() {
    noCanvas();
    createElement('h2', 'Gemini 비전 (로컬 파일 처리)');

    // 1. 로컬 파일 선택 필드
    createElement('p', '**1. 로컬 이미지 파일 선택:**');
    fileInput = createFileInput(handleFile); // 파일 선택 시 handleFile 함수 호출
    fileInput.attribute('accept', 'image/*'); // 이미지 파일만 선택 가능하도록 설정
    
    createElement('p', '**2. 이미지에 대한 질문 입력:**');
    promptField = createInput('이 그림에는 무엇이 있나요?');
    promptField.size(350);

    sendButton = createButton('🖼️ Flask 서버로 이미지 질문 보내기');
    sendButton.mousePressed(askServer);

    responseText = createP('서버의 답변이 여기에 표시됩니다.');
    responseText.style('background-color', '#f0f0f0');
    responseText.style('padding', '15px');
    responseText.style('max-width', '450px');
    responseText.style('border-radius', '5px');
}

// 선택된 파일을 처리하는 함수
function handleFile(file) {
    if (file.type === 'image') {
        responseText.html(`✅ 이미지 파일 **${file.name}** (${file.size} Bytes)이 선택되었습니다. 질문을 입력하고 보내주세요.`);
        // p5.js의 file 객체에 base64 데이터가 이미 포함되어 있습니다.
    } else {
        responseText.html('❌ 이미지 파일만 선택해주세요.');
        fileInput.value(''); // 파일 선택 초기화
    }
}

// Flask 서버에 이미지 질문 요청을 보내는 함수
async function askServer() {
    let file = fileInput.elt.files[0]; // 선택된 파일 객체 가져오기 (p5.js 방식이 아닌 순수 DOM 요소에서)
    let prompt = promptField.value();

    if (!file || prompt === "") {
        responseText.html('❌ 파일을 선택하고 질문을 입력해주세요.');
        return;
    }
    
    // 파일 리더를 사용하여 파일을 Base64로 읽습니다.
    const reader = new FileReader();
    
    reader.onload = async function(e) {
        const base64Image = e.target.result; // Data URL (Base64)
        
        promptField.value('');
        responseText.html(`🌍 Flask 서버를 통해 **${prompt}** (로컬 파일) 요청 중입니다...`);

        // 서버로 보낼 데이터: base64Image와 prompt를 모두 포함
        let dataToSend = { 
            base64_image: base64Image, 
            prompt: prompt 
        };

        try {
            let response = await fetch(serverUrl, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(dataToSend)
            });

            let json = await response.json();

            if (json.success) {
                let reply = json.reply;
                responseText.html(`**🖼️ 이미지 질문 답변:**<br>${reply.replace(/\n/g, '<br>')}`);
            } else {
                responseText.html(`❌ 서버 응답 오류: ${json.error}`);
            }

        } catch (error) {
            responseText.html(`❌ 네트워크 연결 오류. Flask 서버가 실행 중인지 확인하세요: ${error}`);
        }
    };
    
    // 파일을 Data URL(Base64)로 읽기 시작
    reader.readAsDataURL(file); 
}