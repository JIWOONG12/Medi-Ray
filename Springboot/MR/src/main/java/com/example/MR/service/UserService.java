package com.example.MR.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

import com.example.MR.entity.DoctorTb;
import com.example.MR.mapper.MRMapper;

@Service
public class UserService {

    private final MRMapper mapper;
    private final PasswordEncoder passwordEncoder; // PasswordEncoder를 주입받습니다.

    @Autowired
    // 생성자 주입을 통해 MRMapper와 PasswordEncoder를 주입합니다
    public UserService(MRMapper mapper, PasswordEncoder passwordEncoder) {
        this.mapper = mapper;
        this.passwordEncoder = passwordEncoder;
    }

    // 사용자 등록 메서드
    public void registerUser(String DT_CODE, String DT_ID, String DT_PW, String DT_NAME, String DIVISION, String DT_TELNO) {
        // 1. 비밀번호 암호화
        String encodedPassword = passwordEncoder.encode(DT_PW);
        System.out.println("암호화된 비밀번호: " + encodedPassword); // 암호화된 비밀번호 로그 출력

        // 2. DoctorTb 객체 생성 및 데이터 설정 (생성자를 사용하여 초기화)
        DoctorTb doctor = new DoctorTb(DT_CODE, DT_ID, encodedPassword, DT_NAME, DIVISION, DT_TELNO);

        // 3. 데이터베이스에 사용자 등록
        mapper.registDoctor(doctor);
    }

	public void registerUser(DoctorTb dt) {
		this.registerUser(dt.getDT_CODE(), dt.getDT_ID(), dt.getDT_PW(), dt.getDT_NAME(), dt.getDIVISION(), dt.getDT_TELNO());
	}
}
