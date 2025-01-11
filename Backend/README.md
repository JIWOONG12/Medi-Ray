# 📄 BE part 정리  
## 📚 사용 기술
<div> 
  <img src="https://img.shields.io/badge/java-007396?style=for-the-badge&logo=java&logoColor=white"> 
  <img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <br>
  
  <img src="https://img.shields.io/badge/html5-E34F26?style=for-the-badge&logo=html5&logoColor=white">
  <img src="https://img.shields.io/badge/javascript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black">
  <img src="https://img.shields.io/badge/jquery-0769AD?style=for-the-badge&logo=jquery&logoColor=white">
  <br>
  
  <img src="https://img.shields.io/badge/springboot-6DB33F?style=for-the-badge&logo=springboot&logoColor=white"> 
  <img src="https://img.shields.io/badge/fastapi-009688?style=for-the-badge&logo=fastapi&logoColor=white">
  <img src="https://img.shields.io/badge/mariaDB-003545?style=for-the-badge&logo=mariaDB&logoColor=white">
  <br>

  <img src="https://img.shields.io/badge/docker-2496ED?style=for-the-badge&logo=docker&logoColor=white">
  <img src="https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white">
</div>

## 📌 개발환경
+ **Spring Boot**
  + 3.3.4
  + Maven
  + JavaJDK 17
  + Project Metadata 다 기본값(Packaging : Jar)
+ **FastAPI**
  + 0.115.0
  + python 3.10.13 → torch 2.4.1+cpu/ultralytics 8.3.9
+ **MariaDB**
  + 10.5

## ✏️ 시스템 아키텍처 & API 문서
### 시스템 아키텍처
![image](https://github.com/user-attachments/assets/85982069-141e-4ebe-94cc-23ce20919ff2)
![image](https://github.com/user-attachments/assets/873e0903-c74a-431d-bf39-ec71af63fcd8)

### API 문서
 + [API 명세서](https://docs.google.com/spreadsheets/d/1gWSqK_wsTl03aVV3zX7HH4mJWQ9vPX0HMhkupyEhzwc/edit?usp=sharing) 작성중...

## 📝 참고사항
+ Spring Boot
  + frontend 부분 안에 포함하고 있음
+ FastAPI
  + [pytorch_grad_cam](https://github.com/jacobgil/pytorch-grad-cam/tree/master)은 github 에서 다운 받아서 사용
+ DICOM Metadata
  + 기존 데이터에는 **환자코드만 존재**하기 때문에 시현을 위해 이름, ID, 생년월일, 성별을 임의로 지정해줌
