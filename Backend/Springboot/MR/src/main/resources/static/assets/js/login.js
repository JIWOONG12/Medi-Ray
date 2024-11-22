document.addEventListener('DOMContentLoaded', function() {
	// 로그인 버튼 클릭 시
	document.getElementById('loginBtn').addEventListener('click', function() {
		const dtId = document.getElementById('email').value;
		const dtPw = document.getElementById('password').value;


		// 둘다 값이 입력되었는지 확인
		if (dtId === '' || dtPw === '') {
			alert('이메일과 비밀번호를 모두 입력해주세요.');
			return;
		}

		// 서버에 로그인 요청 보내기
		fetch('/login/process', {
			method: 'POST',
			headers: {
				'Content-Type': 'application/x-www-form-urlencoded', // x-www-form-urlencoded 형식 사용
			},
				 body: `username=${encodeURIComponent(dtId)}&password=${encodeURIComponent(dtPw)}` // 필드명을 Spring Security 형식에 맞게 변경
			})
			.then(response => {
            if (response.redirected) {
	                // 로그인 성공 시 리다이렉트
	                window.location.href = "main";
	            } else {
	                return response.json().then(data => {
                    alert(data.message); // 실패 메시지를 alert으로 표시
                });
	            }
	        })
	        .catch(error => {
	            console.error('로그인 처리 중 오류 발생:', error);
	            alert('로그인 중 문제가 발생했습니다. 다시 시도해주세요.');
	        });
	    });
	
	    // 엔터키로 로그인 요청 보내기
	    document.addEventListener('keydown', function(event) {
	        if (event.key === 'Enter') {
	            document.getElementById('loginBtn').click();
	        }
	    });
	    
		// 회원가입 링크 클릭 시
		document.getElementById('signupLink').addEventListener('click', function() {
			window.location.href = 'signup';  // 회원가입 화면으로 이동
		});
		
		// 로고 클릭 시 index.html로 이동
        document.getElementById('logo').addEventListener('click', function () {
          window.location.href = 'index';
        });
});
