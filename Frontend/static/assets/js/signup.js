document.addEventListener('DOMContentLoaded', function() {
	// 전화번호 인증 버튼 클릭 시
	/*document
		.getElementById('verifyPhoneBtn')
		.addEventListener('click', function() {
			alert('인증 완료!');
		});*/

	// 이메일 중복 확인 버튼 클릭 시
	document
		.getElementById('checkEmailBtn')
		.addEventListener('click', function() {
			const email = document.getElementById('email').value;
			const encodedEmail = encodeURIComponent(email);
			const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

			if (!emailRegex.test(email)) {
				alert('이메일 형식이 올바르지 않습니다.');
			} else {
				// AJAX 요청을 통해 이메일 중복 확인
				fetch(`/signup/check-id?dtId=${encodedEmail}`)
					.then(response => response.text())
					.then(data => {
						if (data === '1') {
							alert('이미 등록된 이메일입니다.');
						} else {
							alert('사용 가능한 이메일입니다.');
						}
					})
					.catch(error => {
						console.error('Error:', error);
						alert('이메일 확인 중 오류가 발생했습니다.');
					});
			}
		});

	// 의사 코드 중복 확인 버튼 클릭 시
	document
		.getElementById('checkDoctorCodeBtn')
		.addEventListener('click', function() {
			const doctorCode = document.getElementById('doctor-code').value;
			const encodedDoctorCode = encodeURIComponent(doctorCode);

			// AJAX 요청을 통해 의사 코드 중복 확인
			fetch(`/signup/check-code?dtCode=${encodedDoctorCode}`)
				.then(response => response.json()) // 응답을 JSON으로 처리
				.then(data => {
					if (data === 1) {
						alert('이미 등록된 의사코드입니다.');
					} else if (data === 3) {
						alert('등록할 수 없는 형식의 의사코드입니다.');
					} else if (data === 0) {
						alert('등록 가능한 의사코드입니다.');
					} else {
						alert('의사 코드 확인 중 오류가 발생했습니다.');
						console.log(data);
					}
				})
				.catch(error => {
					console.error('Error:', error);
					alert('의사 코드 확인 중 오류가 발생했습니다.');
				});
		});

	// 비밀번호 유효성 검사 (영문, 숫자, 특수문자 포함, 8자 이상)
	document.getElementById('password').addEventListener('input', function() {
		const password = document.getElementById('password').value;
		const passwordRegex =
			/^(?=.*[a-zA-Z])(?=.*\d)(?=.*[!@#$%^&*(),.?":{}|<>]).{8,}$/;

		if (!passwordRegex.test(password)) {
			document.getElementById('password-message').innerText =
				'비밀번호는 영문, 숫자, 특수문자를 포함하고 8자 이상이어야 합니다.';
		} else {
			document.getElementById('password-message').innerText = ''; // 조건 만족 시 메시지 제거
		}
	});

	// 비밀번호 확인 (일치 여부 확인)
	document
		.getElementById('password-confirm')
		.addEventListener('input', function() {
			const password = document.getElementById('password').value;
			const confirmPassword = document.getElementById('password-confirm').value;
			const confirmPasswordMessage = document.getElementById('confirm-password-message');

			if (password === confirmPassword) {
				confirmPasswordMessage.innerText = '비밀번호가 일치합니다.';
				confirmPasswordMessage.style.color = 'green';  // 일치 시 초록색
			} else {
				confirmPasswordMessage.innerText = '비밀번호가 일치하지 않습니다.';
				confirmPasswordMessage.style.color = 'red';    // 불일치 시 빨간색
			}
		});

	// 완료 버튼 클릭 시 회원가입 처리
	document.getElementById('signupForm').addEventListener('submit', function(event) {
		event.preventDefault(); // 폼 제출 시 기본 동작 방지

		// 입력된 값을 가져옴
		const name = document.getElementById('name').value;
		const phone = document.getElementById('phone').value;
		const email = document.getElementById('email').value;
		const password = document.getElementById('password').value;
		const confirmPassword = document.getElementById('password-confirm').value;
		const doctorCode = document.getElementById('doctor-code').value;
		const hospital = document.getElementById('hospital').value;

		// 필수 입력 필드가 비어 있는지 확인
		if (!name || !phone || !email || !password || !confirmPassword || !doctorCode || !hospital) {
			alert('모든 필수 항목을 입력해 주세요.');
			return; // 입력되지 않은 경우 회원가입 진행 중단
		}

		// 비밀번호와 비밀번호 확인 일치 여부 확인
		if (password !== confirmPassword) {
			alert('비밀번호가 일치하지 않습니다.'); // 비밀번호 불일치 시 알림
			return; // 회원가입 진행 중단
		}


		
		fetch('/signup/register', {
		    method: 'POST',
		    headers: {
		        'Content-Type': 'application/json',
		    },
		    body: JSON.stringify({
				DT_CODE: doctorCode,
			    DT_ID: email,
			    DT_PW: password,
			    DT_NAME: name,
			    DIVISION: hospital,
			    DT_TELNO: phone,
			}),
		})
		.then(response => {
		    console.log('Response status:', response.status); // 응답 상태 코드 확인
		    return response.text();
		})
		.then(data => {
		    if (data === '1') {
		        alert('회원가입이 완료되었습니다!\n로그인을 진행합니다.');
		        window.location.href = 'login';
		    } else if (data === '2') {
		        alert('이메일을 수정해주세요');
		    } else if (data === '3') {
		        alert('의사코드를 수정해주세요.');
		    } else {
		        alert('회원가입에 실패했습니다.');
		    }
		})
		.catch(error => {
		    console.error('Error:', error);
		    alert('회원가입 처리 중 오류가 발생했습니다.');
		});

	});
}); 