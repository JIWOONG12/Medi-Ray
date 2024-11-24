document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('fileInput');
    const selectFilesBtn = document.getElementById('selectFilesBtn');
    const uploadBox = document.getElementById('upload-box');
    const previewContainer = document.getElementById('preview-container');
    const previewImage = document.getElementById('preview-image');
    const closePreviewBtn = document.getElementById('closePreviewBtn');
    const submitBtn = document.getElementById('submitBtn');
    const uploadTitle = document.getElementById('dicomTitle'); // "DICOM Upload" 텍스트 선택
    const patientInfoBody = document.getElementById('patient-info-body');
    let ptCode = null; // 전역 변수로 ptCode 저장
    let fastApiResponse = null; // FastAPI 응답 데이터를 저장할 객체
    let fileName = null;

    // 파일 선택 버튼 클릭 시 파일 입력 창 열기
    selectFilesBtn.addEventListener('click', function(event) {
        event.stopPropagation(); // 중복 호출 방지
        fileInput.click();
    });

    // 파일 선택 시 파일 업로드 처리
    fileInput.addEventListener('change', function(event) {
        const file = event.target.files[0];
        if (file) {
            const formData = new FormData();
            formData.append('file', file); // 'file'은 서버에서 받을 변수명과 일치해야 함

            // jQuery AJAX 요청
            $.ajax({
                url: 'https://localhost:8000/dicom/dupload', // 서버 업로드 엔드포인트
                type: 'POST',
                enctype: 'multipart/form-data',
                data: formData,
                withCredentials: true,
                contentType: false, // multipart/form-data를 사용하기 위해 false로 설정
                processData: false, // 데이터를 쿼리 스트링으로 변환하지 않도록 설정
                success: function(response) {
                    console.log('FastAPI 응답:', response);

                    // ptCode 값을 전역 변수에 저장
                    if (response.metadata && response.metadata.ptCode) {
                        ptCode = response.metadata.ptCode;
                        fastApiResponse = response;
                    }

                    // Base64 이미지 미리보기 설정
                    if (response.image) {
                        previewImage.src = `data:image/png;base64,${response.image}`;
                        previewContainer.classList.remove('hidden');
                        uploadBox.classList.add('hidden');
                        uploadTitle.classList.add('hidden'); // "DICOM Upload" 제목 숨기기
                        fileName = file.name.split('.')[0];
                    }

                    // ptCode를 이용해 담당 의사 정보 가져오기
                    if (response.metadata.ptCode) {
                        console.log(response.metadata.ptCode);
                        $.ajax({
                            url: `/api/upload/dtsearch?ptCode=${response.metadata.ptCode}`,
                            type: 'GET',
                            success: function(doctorResponse) {
                                console.log('Doctor info:', doctorResponse);

                                // 환자 정보를 테이블에 표시
                                patientInfoBody.innerHTML = `
                                    <tr>
                                      <td class="border px-4 py-2">${response.metadata.ptCode}</td>
                                      <td class="border px-4 py-2">${response.metadata.ptName || 'N/A'}</td>
                                      <td class="border px-4 py-2">${response.metadata.ptBirthdate || 'N/A'}</td>
                                      <td class="border px-4 py-2">${response.metadata.ptGen || 'N/A'}</td>
                                      <td class="border px-4 py-2">${doctorResponse.dtName || 'N/A'}</td>
                                    </tr>`;
                            },
                            error: function(xhr, status, error) {
                                console.error('Failed to get doctor info:', xhr, status, error);
                                alert("담당 의사 정보를 가져오는 데 실패했습니다.");
                            }
                        });
                    }
                },
                error: function(xhr, status, error) {
                    console.error('Failed to upload file:', xhr, status, error);
                    alert('파일 업로드에 실패했습니다.');
                }
            });
            event.target.value = ''; // 파일 선택 후 입력 필드 초기화
        }
    });

    // 드래그 앤 드롭 시 파일 업로드 처리
    uploadBox.addEventListener('dragover', function(event) {
        event.preventDefault();
    });

    uploadBox.addEventListener('drop', function(event) {
        event.preventDefault();
        const files = event.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            // 파일 업로드는 change 이벤트를 통해 처리되므로 별도의 함수 호출 불필요
            const event = new Event('change');
            fileInput.dispatchEvent(event);
        }
    });

    // 최종 데이터를 Spring Boot에 업로드
    submitBtn.addEventListener('click', async function () {
        if (!fastApiResponse || !fastApiResponse.metadata) {
            alert('FastAPI 응답 데이터가 없습니다. 파일을 다시 업로드하세요.');
            return;
        }

        // FastAPI 응답 데이터를 활용하여 최종 데이터 생성
        const data = {
            image: fastApiResponse.image, // FastAPI에서 받은 Base64 이미지
            ptCode: fastApiResponse.metadata.ptCode, // FastAPI에서 받은 ptCode
            fileName: fileName, // 미리보기의 파일명
            modelResult: fastApiResponse.model_result
        };
        // 데이터 확인
        /*console.log("Spring Boot로 전송할 데이터:", JSON.stringify(data));*/

        try {
            // Spring Boot 서버에 데이터 전송
            const response = await fetch('/api/upload/imgupload', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            if (!response.ok) {
                const errorResponse = await response.json();
                console.error('Spring Boot 에러 응답:', errorResponse);
                alert('Spring Boot 업로드에 실패했습니다.');
                return;
            }

            const result = await response.json();
            alert('이미지와 결과가 성공적으로 업로드되었습니다.');
            location.reload();
        } catch (error) {
            console.error('Spring Boot 업로드 중 오류:', error);
            alert('Spring Boot 업로드 중 오류가 발생했습니다.');
        }
    });

    // 미리보기 닫기 버튼
    closePreviewBtn.addEventListener('click', function() {
        // 미리보기 숨기기
	    previewContainer.classList.add('hidden');
	    // 업로드 박스 다시 표시
	    uploadBox.classList.remove('hidden');
	    // "DICOM Upload" 제목 다시 표시
	    uploadTitle.classList.remove('hidden');
	    // 미리보기 이미지 초기화
	    previewImage.src = '';
	    // 파일 입력 필드 초기화
	    fileInput.value = '';
	    // 환자 정보 초기화
	    patientInfoBody.innerHTML = '';
	    // 전역 변수 초기화
	    ptCode = null;
	    fastApiResponse = null;
	    fileName = null;
    });

    // 로고 클릭 시 main 페이지로 이동
    document.getElementById('logo').addEventListener('click', function() {
        window.location.href = 'main';
    });
});
