document.addEventListener('DOMContentLoaded', function() {
    const canvas = document.getElementById('diagnosisCanvas');
    const camCanvas = document.getElementById('camCanvas');
    const dicomCanvas = document.getElementById('dicomCanvas');

    const ctx = canvas.getContext('2d');
    const camCtx = camCanvas.getContext('2d'); // CAM 전용 canvas context
    const dicomCtx = dicomCanvas.getContext('2d'); // DICOM 전용 canvas context

    let uploadedImage = null; // DICOM 이미지를 저장할 변수
    let camImage = null; // Grad-CAM 이미지를 저장할 변수


    // 초기 캔버스 크기 저장
    let canvasWidth = canvas.width;
    let canvasHeight = canvas.height;

    // 이전 캔버스 크기 저장
    let prevCanvasWidth = canvasWidth;
    let prevCanvasHeight = canvasHeight;

    // Canvas
    const brushToolButton = document.getElementById('brushTool');
    const rectangleToolButton = document.getElementById('rectangleTool');
    const colorPicker = document.getElementById('colorPicker');
    const brushSize = document.getElementById('brushSize');

    // Cam, dicom
    const imageLoader = document.getElementById('imageLoader');
    const doctorOpinionTextarea = document.querySelector('textarea[placeholder="의사 소견 입력"]');
    const saveButton = document.getElementById('opSubmit'); // 저장 버튼 선택
    const paginationContainer = document.getElementById('pagination'); // 페이지네이션 컨테이너 추가
    const itemsPerPage = 3; // 한 페이지에 표시할 항목 수
    let currentPage = 1;
    let data = []; // 서버에서 가져온 데이터를 저장할 배열

    // CAM 실행할 때 필요한 변수
    let heatmapActive = false; // Grad-CAM 활성화 상태 추적

    // Canvas 실행할 때 필요한 변수
    let isDrawing = false;
    let startX, startY;
    let currentTool = 'brush';
    let selectedColor = '#000000';
    let selectedSize = 5;

    // URL 파라미터에서 ptCode 가져오기
    const urlParams = new URLSearchParams(window.location.search);
    const ptCode = urlParams.get('ptCode');
    let xrayCode = localStorage.getItem('xrayCode');

    // localStorage에 `xrayCode`가 있으면 즉시 의견을 업데이트
    if (xrayCode) {
        fetchOpinionAndUpdate();
        // 페이지가 변경될 때마다 호출 (예: xrayCode가 바뀔 때마다 호출)
        fetchDiagnosisResults(xrayCode);
        const xrayImgPath = `/diagnosis/xray/getImage?ptCode=${encodeURIComponent(ptCode)}&xrayCode=${encodeURIComponent(xrayCode)}`;
        loadImageToCanvas(xrayImgPath);
        refreshDateList();
    } else {
        console.error('localStorage에 xrayCode가 없습니다.');
        redirectToErrorPage(); // 새 페이지로 리다이렉트
    }

    console.log('Loaded xrayCode:', xrayCode); // 받아온 값 확인
    console.log('Received ptCode:', ptCode);
    
    // 새 페이지로 리다이렉트하는 함수
    function redirectToErrorPage() {
        location.reload(); // 현재 페이지 새로고침
    }
    

    // Canvas API
    // 화면 크기에 맞게 캔버스를 리사이즈하는 함수
    function resizeCanvas() {
        scaleCanvasContent();
        const containerWidth = canvas.parentElement.clientWidth;
        const containerHeight = canvas.parentElement.clientHeight;

        // 현재 내용을 저장
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = canvas.width;
        tempCanvas.height = canvas.height;
        tempCanvas.getContext('2d').drawImage(dicomCanvas, 0, 0);

		// HTML 요소의 너비와 높이에 맞추어 캔버스 크기 조정
        // HTML 요소의 너비와 높이에 맞추어 캔버스 크기 조정
        canvas.width = containerWidth;
        canvas.height = containerHeight;
        camCanvas.width = containerWidth;
        camCanvas.height = containerHeight;
        dicomCanvas.width = containerWidth;
        dicomCanvas.height = containerHeight;
    }

    // 창 크기가 변경될 때마다 캔버스를 리사이즈
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas(); // 초기 로딩 시 한 번 호출
    scaleCanvasContent(); // canvas에 그려진 내용 복원

    // Tool selection
    brushToolButton.addEventListener('click', () => (currentTool = 'brush'));
    rectangleToolButton.addEventListener('click', () => (currentTool = 'rectangle'));

    // Update color and size
    colorPicker.addEventListener('input', (e) => (selectedColor = e.target.value));
    brushSize.addEventListener('input', (e) => (selectedSize = e.target.value));

    // Canvas event listeners for drawing
    canvas.addEventListener('mousedown', (e) => {
        isDrawing = true;
        startX = e.offsetX;
        startY = e.offsetY;
    });

    canvas.addEventListener('mousemove', (e) => {
        if (!isDrawing) return;

        if (currentTool === 'brush') {
            ctx.lineWidth = selectedSize;
            ctx.lineCap = 'round';
            ctx.strokeStyle = selectedColor;
            ctx.beginPath();
            ctx.moveTo(startX, startY);
            ctx.lineTo(e.offsetX, e.offsetY);
            ctx.stroke();
            startX = e.offsetX;
            startY = e.offsetY;
        }
    });

    canvas.addEventListener('mouseup', (e) => {
        e.preventDefault();
        isDrawing = false;

        if (currentTool === 'rectangle') {
            const rectWidth = e.offsetX - startX;
            const rectHeight = e.offsetY - startY;
            ctx.strokeStyle = selectedColor;
            ctx.lineWidth = selectedSize;
            ctx.strokeRect(startX, startY, rectWidth, rectHeight);
        }
    });

    // Clear only drawings on the canvas, keep uploaded image
    // Clear 버튼 기능: 드로잉 및 CAM만 지우고 DICOM 이미지는 유지
    document.getElementById('clearCanvas').addEventListener('click', () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        resetToolSelection()
        drawDicomImage(); // Reset DICOM image on clear
        ctx.restore();
    });
    

    // CAM button functionality
    // Grad-CAM 버튼 클릭 이벤트
    document.getElementById('camButton').addEventListener('click', () => {
        heatmapActive = !heatmapActive; // 현재 Grad-CAM 활성 상태를 반전

		keepcamToolSelection(); // 버튼을 체크 상태로 유지
        if (heatmapActive) {
            console.log("Grad-CAM 활성화: Canvas 데이터를 Base64로 변환 중...");
	        

            // Canvas 데이터를 Base64로 변환
            const base64Img = dicomCanvas.toDataURL("image/png").split(",")[1]; // Base64 인코딩, 헤더 제거
            

            // AJAX 요청을 통해 FastAPI로 Base64 데이터 전송
            $.ajax({
                url: 'https://localhost:8000/dicom/resnet_gradcam', // FastAPI 엔드포인트
                type: 'POST', // POST 요청
                contentType: 'application/json', // JSON 형식으로 전송
                data: JSON.stringify({ base64_img: base64Img }), // 요청 본문에 Base64 데이터 포함
                success: function (response) {
                    console.log("FastAPI 응답 데이터:", response); // FastAPI 응답 로그
                    // 서버 응답에서 Base64 이미지 데이터 처리
                    if (response.image) {
                        const img = new Image();
                        img.src = `data:image/png;base64,${response.image}`; // Base64 데이터를 이미지 소스로 설정

                        img.onload = () => {
                            // Grad-CAM 이미지를 캔버스에 렌더링
                            camImage = img; // camImage에 이미지 저장
                            drawCamImage(); // 이미지를 그리는 함수 호출
                        };
                    } else {
                        console.error("FastAPI 응답에 이미지 데이터가 없습니다.");
                        alert("Grad-CAM 결과를 가져오지 못했습니다.");
                    }
                },
                error: function (xhr, status, error) {
                    console.error("Grad-CAM 요청 중 오류 발생:", error);
                    console.error("XHR 응답:", xhr.responseText);
                    alert("Grad-CAM 생성 중 오류가 발생했습니다."); // 사용자에게 오류 알림
                },
            });
        } else {
            console.log("Grad-CAM 비활성화: 원본 이미지 표시");
            // Grad-CAM 비활성화: 원본 이미지 표시
            camImage = null; // camImage 초기화
            camCtx.clearRect(0, 0, camCanvas.width, camCanvas.height); // Grad-CAM 캔버스 초기화
            resetcamToolSelection()
        }
    });
    

    // 툴 버튼들의 선택 상태를 해제하는 함수
    function resetToolSelection() {
        const toolInputs = document.querySelectorAll('input[name="tool"]');
        toolInputs.forEach(input => {
            input.checked = false;
        });
    }
    // cam 툴 버튼들의 선택 상태를 해제하는 함수
    function resetcamToolSelection() {
        const toolInputs = document.querySelectorAll('input[name="camtool"]');
        toolInputs.forEach(input => {
            input.checked = false;
        });
    }
    // cam 툴 버튼들의 선택 상태를 유지하는 함수
    function keepcamToolSelection() {
        const toolInputs = document.querySelectorAll('input[name="camtool"]');
        toolInputs.forEach(input => {
            input.checked = true;
            console.log(`Tool ${input.id} unchecked`); // 상태 확인 로그
        });
    }

    // 촬영 리스트 데이터 가져오기
    fetch(`/diagnosis/xray/dateList?ptCode=${encodeURIComponent(ptCode)}`)
        .then(response => response.json())
        .then(fetchedData => {
            data = fetchedData; // 데이터 저장
            renderPage(); // 첫 페이지 렌더링
            renderPagination(); // 페이지네이션 렌더링
        })
        .catch(error => {
            console.error('Error fetching dateList:', error);
        });
        

    // 페이지 렌더링 함수
    function renderPage() {
        const startIndex = (currentPage - 1) * itemsPerPage;
        const endIndex = startIndex + itemsPerPage;
        const itemsToDisplay = data.slice(startIndex, endIndex);
        
        // 모든 라디오 버튼 초기화
	    const allRadioButtons = document.querySelectorAll('#patientList input[type="radio"]');
	    allRadioButtons.forEach(radio => {
	        radio.checked = false;
	    });

        document.querySelectorAll('#patientList > li > label').forEach(item => {
            item.innerText = "";
            item.classList.remove('text-blue-800', 'font-bold');
        	item.classList.add('text-blue-500');
        });

		// 현재 xrayCode찾기
		const currentXrayCode = localStorage.getItem('xrayCode');

        // 촬영리스트에 값 할당하기
        itemsToDisplay.forEach((item, index) => {
            const input = document.getElementById(`patientBtn${index}`);
	        const label = document.querySelector(`label[for="patientBtn${index}"]`);
	        label.innerText = item.xrayDate;
	        input.value = item.xrayCode;
	        
	        // 현재 xrayCode와 일치하는 항목 강조
	        if (item.xrayCode === currentXrayCode) {
	            input.checked = true;
	            label.classList.remove('text-blue-500');
	            label.classList.add('text-blue-800', 'font-bold');
	        }

			
            // 입력 요소의 change 이벤트 - AJAX를 사용해 데이터 전송
	        input.onchange = function() {
	            if (input.checked) {
	                // 사진 리스트 버튼 클릭 시 툴 버튼들의 선택 상태 해제
	                resetToolSelection();
	                resetcamToolSelection();
	                
	
	                fetch(`/diagnosis/xray/imgDate?ptCode=${encodeURIComponent(ptCode)}&xrayDate=${encodeURIComponent(item.xrayDate)}`)
	                    .then(response => response.json())
	                    .then(imgData => {
	                        if (imgData.length > 0) {
								xrayCode = imgData[0].xrayCode;  // xrayCode 직접 설정
	                            updateScreen(imgData); // 화면에 데이터 업데이트
	
	                            // 선택된 xrayCode에 맞는 이미지와 의견 업데이트
	                            fetchOpinionAndUpdate(); // 의견 업데이트
	                            // 페이지가 변경될 때마다 호출 (예: xrayCode가 바뀔 때마다 호출)
	                            fetchDiagnosisResults(xrayCode);
	                            // 모든 레이블 스타일 초기화
	                            document.querySelectorAll('#patientList > li > label').forEach(l => {
	                                l.classList.remove('text-blue-800', 'font-bold');
	                                l.classList.add('text-blue-500');
	                            });
	
	                            // xrayCode에 맞는 리스트 선택되게 하기
	                            // 선택된 항목만 강조
	                            label.classList.remove('text-blue-500');
	                            label.classList.add('text-blue-800', 'font-bold');

	                        } else {
	                            alert('선택한 데이터가 존재하지 않습니다.');
	                        }
	                    })
	                    .catch(error => {
	                        console.error('Error fetching imgDate data:', error);
	                        refreshDateList(); // 오류가 발생하면 리스트 새로고침
	                    });
	            }
	        };
	    });
	}

    // 이미지 로딩 후 캔버스에 표시
    function loadImageToCanvas(imgUrl) {
        const img = new Image();
        img.crossOrigin = "anonymous"; // CORS 설정 (서버에서 CORS가 허용되는 경우에만 필요)
        img.onload = () => {
            uploadedImage = img;
            drawDicomImage(); // 이미지 다시 그리기
            drawCamImage();
            adjustCanvasSize(); // 이미지 로드 후 canvas 크기 조정
        };
        img.onerror = () => {
        console.error('이미지를 불러올 수 없습니다.');
        alert('선택한 이미지가 삭제되었습니다. 리스트를 새로고침합니다.');
        refreshDateList(); // 리스트 새로고침
    	};
        img.src = imgUrl; // 이미지 경로 설정
    }

    // DICOM 비율에 맞춰 Canvas 크기 조정
    function adjustCanvasSize() {
        if (uploadedImage) {
            const aspectRatio = uploadedImage.width / uploadedImage.height;

            // 이전 캔버스 크기 저장
            prevCanvasWidth = canvasWidth;
            prevCanvasHeight = canvasHeight;

            // DICOM Canvas 비율 조정
            dicomCanvas.width = dicomCanvas.parentElement.clientWidth;
            dicomCanvas.height = dicomCanvas.width / aspectRatio;

            // 다른 캔버스 크기도 동기화
            canvas.width = dicomCanvas.width;
            canvas.height = dicomCanvas.height;

            camCanvas.width = dicomCanvas.width;
            camCanvas.height = dicomCanvas.height;

            // 새로운 캔버스 크기 저장
            canvasWidth = canvas.width;
            canvasHeight = canvas.height;

            // DICOM 이미지를 먼저 그리고 기존 그림을 복원
            drawDicomImage();
            
            // canvas에 그려진 내용 복원
        	scaleCanvasContent();
        	
        	
        }
    }

    // 기존의 그린 내용을 스케일링하여 복원
    function scaleCanvasContent() {
        // 이전 크기의 캔버스 내용을 이미지로 저장
        const savedDrawing = new Image();
        savedDrawing.src = canvas.toDataURL();

        savedDrawing.onload = function() {
            ctx.save();
            ctx.scale(canvasWidth / prevCanvasWidth, canvasHeight / prevCanvasHeight);
            ctx.drawImage(savedDrawing, 0, 0);
            ctx.restore();
        };
    }
	
	// dicomcanvas에 그려진 내용을 스케일링하여 다시 그려줌
    function drawDicomImage() {
        if (uploadedImage) {
            const scaleFactor = Math.min(
                dicomCanvas.width / uploadedImage.width,
                dicomCanvas.height / uploadedImage.height
            );
            const newWidth = uploadedImage.width * scaleFactor;
            const newHeight = uploadedImage.height * scaleFactor;
            const offsetX = (dicomCanvas.width - newWidth) / 2;
            const offsetY = (dicomCanvas.height - newHeight) / 2;

            dicomCtx.clearRect(0, 0, dicomCanvas.width, dicomCanvas.height);
            dicomCtx.drawImage(uploadedImage, offsetX, offsetY, newWidth, newHeight);
        }
    }
    
    // camImage를 camCanvas에 그리는 함수
    function drawCamImage() {
	    if (camImage) {
	        const scaleFactor = Math.min(
	            camCanvas.width / camImage.width,
	            camCanvas.height / camImage.height
	        );
	        const newWidth = camImage.width * scaleFactor;
	        const newHeight = camImage.height * scaleFactor;
	        const offsetX = (camCanvas.width - newWidth) / 2;
	        const offsetY = (camCanvas.height - newHeight) / 2;
	
	        camCtx.clearRect(0, 0, camCanvas.width, camCanvas.height);
	        camCtx.drawImage(camImage, offsetX, offsetY, newWidth, newHeight);
	    }
	}

    

    // 창 크기 변경 시 리사이즈와 복원 함수 호출
    window.addEventListener('resize', function(){
		adjustCanvasSize();
		scaleCanvasContent();
		drawCamImage();
		refreshDateList();
		});

    // 초기 설정
    adjustCanvasSize();

    // 화면 업데이트 함수
    function updateScreen(imgData) {
        if (imgData.length > 0) {
            xrayCode = imgData[0].xrayCode;
            localStorage.setItem('xrayCode', xrayCode); // localStorage에 저장
            console.log('Updated xrayCode:', xrayCode);

            // 이미지 경로가 있으면 Canvas에 이미지를 로드
            const xrayImgPath = `/diagnosis/xray/getImage?ptCode=${encodeURIComponent(ptCode)}&xrayCode=${encodeURIComponent(xrayCode)}`;
            loadImageToCanvas(xrayImgPath); // HTTP URL로 변환된 경로 사용

            // xrayCode를 사용해 서버에서 데이터 가져오기
            fetchOpinionAndUpdate();
            fetchDiagnosisResults(xrayCode);
            refreshDateList();
        }
    }

    // 이미지 업로드 이벤트 (input 파일 선택 시 캔버스에 표시)
    imageLoader.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (event) => loadImageToCanvas(event.target.result);
        reader.readAsDataURL(file);
    });

    window.uploadedImage = uploadedImage; // 전역으로 설정하여 다른 스크립트에서 접근 가능

    // 페이지네이션 렌더링 함수
    function renderPagination() {
        paginationContainer.innerHTML = ''; // 기존 페이지네이션 내용 지우기

        const totalPages = Math.ceil(data.length / itemsPerPage);

        for (let i = 1; i <= totalPages; i++) {
            const pageButton = document.createElement('button');
            pageButton.innerText = i;
            pageButton.classList.add('px-3', 'py-1', 'mx-1', 'bg-gray-200', 'hover:bg-gray-300', 'rounded');
            if (i === currentPage) {
                pageButton.classList.add('bg-blue-500', 'text-white'); // 현재 페이지 강조
            }
            pageButton.addEventListener('click', function() {
                currentPage = i;
                renderPage();
                renderPagination();
            });
            paginationContainer.appendChild(pageButton);
        }
    }

    // xrayCode에 맞는 진단 결과를 불러와서 화면에 표시하는 함수
    function fetchDiagnosisResults(xrayCode) {
        const diagnosisResults = document.getElementById('diagnosisResults');
        const table = diagnosisResults.querySelector('table');
        const maxLiCount = 4;

        fetch(`/diagnosis/xray/result?xrayCode=${encodeURIComponent(xrayCode)}`)
            .then(response => response.json())
            .then(data => {
                if (data.length === 0) {
                    data = [{ name: "Normal", value: " " }];
                } else {
                    data = data.map(item => ({
                        name: item.labelName || "&nbsp;",
                        value: `${item.probability?.toFixed(1) || "&nbsp;"}`
                    }));
                }

                // 데이터가 4개 미만일 경우 공백으로 채우기
                while (data.length < maxLiCount) {
                    data.push({ name: "&nbsp;", value: "&nbsp;" });
                }

                // 테이블 초기화
                table.innerHTML = "";

                // 데이터 추가
                data.forEach(result => {
                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td class="w-1/2 text-left p-2">${result.name}</td>
                        <td class="w-1/2 text-right p-2 font-semibold">${result.value}</td>
                    `;
                    table.appendChild(tr);
                });
            })
            .catch(error => console.error('Error fetching label probabilities:', error));
    }

    // 서버에서 xrayCode로 데이터 가져오기
    function fetchOpinionAndUpdate() {
        if (xrayCode) {
            fetch(`/diagnosis/xray/dtOpinion?xrayCode=${encodeURIComponent(xrayCode)}`)
                .then(response => {
                    if (!response.ok) {
                        throw new Error('의견 데이터를 가져오는 데 실패했습니다.');
                    }
                    return response.text(); // 텍스트 형식으로 반환받음
                })
                .then(data => {
                    console.log("Received data from server:", data);
                    doctorOpinionTextarea.value = data || "";
                })
                .catch(error => {
                    console.error('의사 소견 데이터를 가져오는 중 오류 발생:', error);
                });
        } else {
            console.error('xrayCode가 없습니다. 다시 확인하세요.');
        }
    }

    // 전송 버튼 클릭 시 의견 업데이트
    saveButton.addEventListener('click', function() {
        const updatedOpinion = doctorOpinionTextarea.value;

        if (xrayCode && updatedOpinion) {
            fetch(`/diagnosis/xray/doUpload`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: new URLSearchParams({
                    xrayCode: xrayCode,
                    dtOpinion: updatedOpinion,
                }),
            })
                .then(response => {
                    if (!response.ok) {
                        throw new Error('의견 업데이트에 실패했습니다.');
                    }
                    return response.text();
                })
                .then(message => {
                    alert(message);
                })
                .catch(error => {
                    console.error('의견 업데이트 중 오류 발생:', error);
                    alert('의견 업데이트 중 오류가 발생했습니다.');
                });
        } else {
            alert('의사 소견을 입력해주세요.');
        }
    });

    // 새로고침 함수
    function refreshDateList(){
	    fetch(`/diagnosis/xray/dateList?ptCode=${encodeURIComponent(ptCode)}`)
	        .then(response => response.json())
	        .then(fetchedData => {
	            data = fetchedData; // 최신 데이터를 저장
	            renderPage(); // 페이지 업데이트
	            renderPagination(); // 페이지네이션 업데이트
	        })
	        .catch(error => {
	        	console.error('Error fetching dateList:', error);
				redirectToErrorPage();
			});
	}
});
