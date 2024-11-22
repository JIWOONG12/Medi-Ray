// 사용자 정보 UI 업데이트 함수
function fetchUserName() {
  fetch('/user/get-name')
    .then(response => {
      if (!response.ok) {
        throw new Error('사용자 정보를 가져오지 못했습니다.');
      }
      return response.text();
    })
    .then(name => {
      if (name !== '사용자 없음') {
        updateUserInfo(name); // 사용자 이름 UI 업데이트
      } else {
        updateUserInfo(null); // 이름이 없을 경우 처리
      }
    })
    .catch(error => {
      console.error('Error fetching user name:', error);
    });
}

// 사용자 이름을 UI에 업데이트
function updateUserInfo(name) {
  const userInfoContainer = document.getElementById('user-info');
  userInfoContainer.innerHTML = ''; // 기존 내용 초기화

  if (name) {
    const userNameSpan = document.createElement('span');
    userNameSpan.textContent = `${name} 님`;
    userInfoContainer.appendChild(userNameSpan);
  } else {
    userInfoContainer.textContent = '로그인 해주세요';
  }
}

// 페이지 로드 시 실행
document.addEventListener('DOMContentLoaded', function () {
  fetchUserName(); // 페이지 로드 시 로그인된 사용자 dtName 가져오기
});

