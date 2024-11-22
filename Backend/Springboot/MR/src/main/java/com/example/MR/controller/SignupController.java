package com.example.MR.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.bind.annotation.RestController;

import com.example.MR.entity.DoctorTb;
import com.example.MR.mapper.MRMapper;
import com.example.MR.service.UserService;

@RestController
@RequestMapping("/signup")
public class SignupController {

	@Autowired
	private MRMapper mapper;
	
	@Autowired
	private UserService service;
	
	
	// 아이디 중복확인
	@GetMapping("/check-id")
	public int checkid(@RequestParam("dtId") String DT_ID) {
		if(mapper.checkDuplicateId(DT_ID) != null) {
			return 1;
		};
		return 0;
	}
	
	// 의사코드 중복확인
	@GetMapping("/check-code")
	@ResponseBody // JSON 형식으로 반환하도록 설정
	public int checkcode(@RequestParam("dtCode") String dtCode) {
	    try {
	        // 중복 검사
	        if (mapper.checkDuplicateCode(dtCode) != null) {
	            return 1; // 중복된 경우 1 반환
	        }
	        return 0; // 사용 가능한 경우 0 반환

	    } catch (NumberFormatException e) {
	        return 3; // int로 변환할 수 없는 경우 3 반환
	    }
	}

	
	// 회원가입 완료
	@PostMapping("/register")
	public int registerDoctor(@RequestBody DoctorTb dt) {
	    try {
	    	// 이메일 중복 확인
            if (mapper.checkDuplicateId(dt.getDT_ID()) != null) {
                return 2;  // 이메일 중복 시 2 반환
            }
            // 의사 코드 중복 확인
            if (mapper.checkDuplicateCode(dt.getDT_CODE()) != null) {
                return 3;  // 의사 코드 중복 시 3 반환
            }

            System.out.println(dt);
	        // 중복이 없는 경우 회원 정보를 DB에 삽입
//	        mapper.registDoctor(dt);
            service.registerUser(dt);
	        return 1; // 성공 시 1 반환
	    } catch (Exception e) {
	        e.printStackTrace(); // 에러가 발생하면 출력
	        return 0; // 실패 시 0 반환
	    }
	}
}
