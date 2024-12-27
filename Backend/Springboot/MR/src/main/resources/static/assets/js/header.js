document.addEventListener('DOMContentLoaded', function () {
  // 서버에서 사용자 정보 가져오기
  function fetchUserName() {
    return fetch('/user/mypage', {
      method: 'GET',
      credentials: 'include'  // 세션 기반 인증을 위한 설정
    })
    .then(response => {
      if (!response.ok) {
        throw new Error('사용자 정보를 가져오지 못했습니다.');
      }
      return response.json(); // JSON 데이터를 반환하도록 수정
    })
    .then(doctorDetails => {
      console.log('Fetched user name:', doctorDetails.DT_NAME);  // 서버에서 가져온 사용자 이름 출력
      const userName = doctorDetails.DT_NAME; // DT_NAME만 추출
      return userName;
    })
    .catch(error => {
      console.error('Error fetching user name:', error);
      return null;  // 에러가 발생하면 null 반환
    });
  }

  // 로그인/로그아웃 UI를 동적으로 업데이트하는 함수
  function updateAuthUI(userName) {
	console.log('Updating UI with name:', userName);  // updateAuthUI 함수에 전달된 이름 출력
    const authContainer = document.getElementById('user-info');
    authContainer.innerHTML = ''; // 기존 내용 초기화

    if (userName) {
      // 로그인된 상태: 사용자 이름과 로그아웃 버튼 표시
      const userNameSpan = document.createElement('span');
      userNameSpan.textContent = `${userName} 님`;
      userNameSpan.style.cursor = 'pointer';  // 사용자 이름에 클릭 가능 포인터 표시

      // 사용자 이름 클릭 시 마이페이지로 이동
      userNameSpan.addEventListener('click', function () {
        window.location.href = '/mypage';  // 마이페이지로 이동
      });

      const logoutButton = document.createElement('button');
      logoutButton.textContent = '로그아웃';
      logoutButton.classList.add('text-red-500', 'text-lg',);
      logoutButton.addEventListener('click', function() {
	    fetch('/logout', { method: 'POST' })
	        .then(() => {
	            alert('로그아웃되었습니다.');
	            window.location.href = '/login'; // 로그아웃 후 로그인 페이지로 이동
	        })
	        .catch(error => {
	            console.error('로그아웃 중 오류 발생:', error);
	            alert('로그아웃 중 문제가 발생했습니다. 다시 시도해주세요.');
	        });
	});


      authContainer.appendChild(userNameSpan);
      authContainer.appendChild(logoutButton);
    } else {
      // 비로그인 상태: 로그인 버튼 표시
      const loginButton = document.createElement('button');
      loginButton.textContent = '로그인';
      loginButton.classList.add('text-blue-500', 'text-lg', 'ml-4');
      loginButton.addEventListener('click', function () {
        alert('로그인 페이지로 이동합니다.');
        window.location.href = '/login'; // 실제 로그인 페이지 경로로 수정
      });

      authContainer.appendChild(loginButton);
    }
  }
  
  // 로고 클릭 시 main으로 이동
  document.getElementById('logo').addEventListener('click', function () {
    // 현재 경로 확인
    const currentPath = window.location.pathname;

    // xray.html일 경우 경로를 "../main"으로 변경
    if (currentPath.includes('xray')) {
      window.location.href = '../main';
    } else {
      window.location.href = 'main';
    }
  });

  // 아코디언 메뉴 동작
  const menuToggle = document.getElementById('menuToggle');
  const accordionMenu = document.getElementById('accordionMenu');
  const closeBtn = document.getElementById('closeBtn');

  menuToggle.addEventListener('click', function () {
    accordionMenu.classList.toggle('open');
  });

  closeBtn.addEventListener('click', function () {
    accordionMenu.classList.toggle('open');
  });

  // 페이지 로드 시 사용자 정보를 가져와 UI 업데이트
  fetchUserName().then(name => {
    if (name !== null) {
      updateAuthUI(name);  // 이름이 있을 경우 UI 업데이트
    } else {
      updateAuthUI(null);  // 로그인되지 않은 경우
    }
  });
  
});
