document.addEventListener('DOMContentLoaded', function() {
	const searchBtn = document.getElementById('searchBtn');
	const patientTableBody = document.getElementById('patientTableBody');
	const userInfo = document.getElementById('user-info');
	const backBtn = document.getElementById('backBtn');

	// 사용자 이름 (로그인 상태라고 가정)
	const loggedInUser = '신속주 님';

	// 로그인 정보 표시
	renderUserInfo();

	document.body.addEventListener('click', function(event) {
		// 특정 조건에 맞는 요소에만 이벤트가 발생하도록 설정
		if (event.target.matches('#patientTableBody a')) {
			event.preventDefault(); // 기본 동작(페이지 이동) 방지
			const xrayCode = event.target.getAttribute('data-xray-code');
			const ptCode = event.target.getAttribute('data-pt-code');
			console.log('X-ray Code:', xrayCode);
			console.log('PT Code:', ptCode);

			// 원하는 작업 수행
			storeXrayCode(xrayCode);
			window.location.href = "/diagnosis/xray?ptCode=" + ptCode;
		}
	});

	// 엔터누르면 서버에 요청 보내기	
	document.getElementById('inputValue').addEventListener('keydown', function(event) {
		if (event.key === 'Enter') {
			// 특정 컴포넌트의 클릭 이벤트 트리거
			document.getElementById('searchBtn').click();
		}
	})
	searchBtn.addEventListener('click', function() {
		const inputValue = document.getElementById('inputValue').value.trim();

		if (inputValue === '') {
			alert('환자 이름을 입력해주세요.');
			return;
		}

		const url = `/diagnosis/search?inputValue=${encodeURIComponent(inputValue)}`;
		// console.log(url);

		// 서버로부터 데이터를 가져오는 부분
		fetch(url)
			.then(response => {
				if (!response.ok) {
					throw new Error('서버 응답 오류');
				}
				return response.json();
			})
			.then(data => {
				console.log('Fetched data:', data); // 전체 데이터 구조 확인
				renderPatientTable(data); // 받아온 데이터를 테이블에 렌더링
			})
			.catch(error => {
				console.error('데이터 가져오기 중 오류 발생:', error);
				alert('환자 데이터를 가져오는 중 오류가 발생했습니다.');
			});
	});

	// 테이블 데이터 렌더링 함수
	function renderPatientTable(patients) {
		patientTableBody.innerHTML = ''; // 기존 데이터 초기화

		if (patients.length === 0) {
			patientTableBody.innerHTML = `<tr><td colspan="5" class="border px-4 py-2">검색 결과가 없습니다</td></tr>`;
		} else {
			patients.forEach((patient) => {
				const row = document.createElement('tr');
				/*여기서 어떻게 url을 넘어 갈 것인지 설정해줘야함*/
				//              <a href="diagnosis/xray?ptCode=${encodeURIComponent(patient.ptCode)}" class="text-blue-500 hover:underline" onclick="javascript:storeXrayCode('${patient.xrayCode}')">${patient.ptName}</a>
				row.innerHTML = `
            <td class="border px-4 py-2">${patient.ptCode}</td>
            <td class="border px-4 py-2">
              <a href="#" class="text-blue-500 hover:underline" data-pt-code="${encodeURIComponent(patient.ptCode)}" data-xray-code="${patient.xrayCode}">${patient.ptName}</a>
            </td>
            <td class="border px-4 py-2">${patient.ptBirthdate}</td>
            <td class="border px-4 py-2">${patient.ptGen}</td>
            <td class="border px-4 py-2">${patient.dtName}</td>
          `;
				patientTableBody.appendChild(row);
			});
		}
	}

	// xrayCode를 localStorage에 저장하는 함수
	function storeXrayCode(xrayCode) {
		localStorage.setItem('xrayCode', xrayCode);
	}

	// 로그인 정보 표시 함수
	function renderUserInfo() {
		const userHtml = `
        <i class="fas fa-user-circle text-2xl"></i>
        <span class="text-lg font-semibold">${loggedInUser}</span>
        <button class="text-red-500 text-lg" id="logoutBtn">로그아웃</button>
      `;
		userInfo.innerHTML = userHtml;

		// 로그아웃 버튼 클릭 이벤트
		const logoutBtn = document.getElementById('logoutBtn');
		logoutBtn.addEventListener('click', function() {
			alert('로그아웃 되었습니다.');
			window.location.href = '/login'; // 로그인 페이지로 이동
		});
	}

	// 뒤로가기 버튼 클릭 시
	if (backBtn) {
		backBtn.addEventListener('click', function() {
			window.history.back(); // 이전 페이지로 이동
		});
	}
});
