// 로그인 버튼 클릭 시 로그인 페이지로 이동
document
  .getElementById('loginBtn')
  .addEventListener('click', function () {
    window.location.href = 'login'; // '/login'을 실제 로그인 페이지 경로로 변경
  });

// 회원가입 버튼 클릭 시 회원가입 페이지로 이동
document
  .getElementById('signUpBtn')
  .addEventListener('click', function () {
    window.location.href = 'signup'; // '/signup'을 실제 회원가입 페이지 경로로 변경
  });

// 로고 클릭 시 index.html로 이동 
document.getElementById('logo').addEventListener('click', function () {
  window.location.href = 'index';
});

// 아코디언 메뉴 열기/닫기
const menuToggle = document.getElementById('menuToggle');
const accordionMenu = document.getElementById('accordionMenu');
const closeBtn = document.getElementById('closeBtn');

if (menuToggle) {
  menuToggle.addEventListener('click', () => {
    accordionMenu.classList.add('open'); // 메뉴 열기
  });

  closeBtn.addEventListener('click', () => {
    accordionMenu.classList.remove('open'); // 메뉴 닫기
  });
}