package com.example.MR.controller;

import java.util.ArrayList;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import com.example.MR.entity.DoctorTb;
import com.example.MR.mapper.MRMapper;

import jakarta.servlet.http.HttpSession;

@RestController
@RequestMapping("/login")
public class LoginController {
	
	@Autowired
    private MRMapper mapper;
	
	// 로그인
    @PostMapping("/search")
    public int diagnosis(@RequestParam("dtId") String dtId, @RequestParam("dtPw") String dtPw, HttpSession session) {
    	DoctorTb doctor = mapper.validateLogin(dtId, dtPw);
    	if(doctor != null) {
    		// 로그인 성공 시, 세션에 dtId 저장
            session.setAttribute("dtId", doctor.getDtId());
    		return 1; // 로그인 성공
    	};
    	return 0; // 로그인 실패
    }
	
}
