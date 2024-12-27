<<<<<<< HEAD
# MediRay
인공지능사관학교 최종성과물 발표회
=======
# 📘 Medi-Ray 프로젝트 소개
### 📢 주제 및 목표
+ 주제 : **ViT(Vision Transformer;비전 트랜스포머)를 활용한 흉부 방사선 폐 진단 서비스**
+ 목표
  + **ViT** 모델을 이용한 높은 효율성
  + **Dicom 형식의 대규모 공개 데이터셋(MIMIC-CXR Database v2.1.0 4.7TB) 활용**으로 데이터 투명성 제공
  + **Multi-Label Classification**을 통한 최대 14가지 비정상 소견을 진단 및 보조
  + 의사의 편의성 향상을 위해  **Canvas API**로 흉부 x-ray에 직접 붓 그림이나 사각형을 그릴 수 있는 기능 제공
### 📆 프로젝트 기간 
+ 계획/분석/설계 : 24.09.23 - 24.10.18
+ 구현 : 24.10.18 - 24.11.25
### ⛰️ 결과
+ 인공지능사관학교 최종 성과물 발표 장려상 수상  
+ 2024 AI 활용 사회문제 해결 아이디어 공모전 우수상 수상
  
→ **전체 프로젝트에 대한 설명 github 주소** : [Medi-Ray](https://github.com/JIWOONG12/Medi-Ray)  

    
## 📄 BE part 정리  
### 📚 사용 기술
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

### 📌 개발환경
+ **GPU**
  + Pytorch 2.3.1(Python 3.10.13, CUDA 11.8, cuDNN 8.9.7)
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

## 📝 주요 업무
+ **Backend 개발**
  + Spring Boot와 MariaDB, FastAPI를 연동하고 데이터 관리 로직 설계 및 구현
  + Spring Security를 이용하여 인증되지 않은 사용자 접근 불가하도록 설계
+ **데이터 관리**
  + HeidiSQL을 이용해 데이터베이스 테이블 관리 및 CRUD 기능 개발
+ **프론트 기능 개선**
  + CanvasAPI 랜더링 문제 해결
>>>>>>> 5fb1e33df6d5ef8200ff2c743dba0908454dce90
