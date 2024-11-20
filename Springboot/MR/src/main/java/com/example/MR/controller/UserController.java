package com.example.MR.controller;

import java.util.Map;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.example.MR.entity.DoctorTb;
import com.example.MR.mapper.MRMapper;


@RestController
@RequestMapping("/user")
public class UserController {
	
	@Autowired
    private MRMapper mapper;
	private final PasswordEncoder passwordEncoder;
	
	public UserController(MRMapper mapper, PasswordEncoder passwordEncoder) {
        this.mapper = mapper;
        this.passwordEncoder = passwordEncoder;
    }
	
	
	
	// 로그인 후 사용자 이름 가져오기
    @GetMapping("/get-name")
    public ResponseEntity<String> getName(@AuthenticationPrincipal UserDetails userDetails) {
        String dtId = userDetails.getUsername();  // Spring Security에서 인증된 사용자 ID 가져오기

        // dtId로 데이터베이스에서 정보 조회
        DoctorTb doctor = mapper.checkDuplicateId(dtId);
        
        if (doctor != null) {
            return ResponseEntity.ok(doctor.getDT_NAME());  // dtName만 반환
        } else {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body("해당 사용자를 찾을 수 없습니다.");
        }
    }
    
    // 마이페이지(정보 불러오기)
    @GetMapping("/mypage")
    public ResponseEntity<DoctorTb> getUserDetails(@AuthenticationPrincipal UserDetails userDetails) {
    	String dtId = userDetails.getUsername();  // 인증된 사용자 ID 가져오기
    	
    	DoctorTb doctorDetails = mapper.checkDuplicateId(dtId);
        if (doctorDetails != null) {
            return ResponseEntity.ok(doctorDetails);  // 첫 번째 의사 정보 반환
        }
        return ResponseEntity.status(HttpStatus.NOT_FOUND).build();
    }
    
    // 마이페이지(비밀번호 변경)
    @PutMapping("/changepw")
    public ResponseEntity<Integer> changePassword(@RequestBody Map<String, String> requestData, Authentication authentication) {
    	String currentPw = requestData.get("currentPw");  // 현재 비밀번호
        String newPw = requestData.get("newPw");          // 새 비밀번호
        String dtId = authentication.getName();           // 인증된 사용자 ID 가져오기
        
        DoctorTb doctor = mapper.checkDuplicateId(dtId);
        
        // 2. 현재 비밀번호가 일치하는지 확인
        if (!passwordEncoder.matches(currentPw, doctor.getDT_PW())) {
            return ResponseEntity.status(HttpStatus.FORBIDDEN).body(-1); // 비밀번호 불일치
        }

        // 3. 새 비밀번호 암호화 후 업데이트
        String encodedNewPw = passwordEncoder.encode(newPw);
        try {
            mapper.updateDoctorPw(dtId, encodedNewPw); // 암호화된 새 비밀번호로 업데이트
            return ResponseEntity.ok(1);  // 성공 시 1 반환
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(0);  // 실패 시 0 반환
        }
    }
    
    
    // 마이페이지(의사소속 변경)
    @PutMapping("/changedv")
    public ResponseEntity<Integer> changeDivision(@RequestBody Map<String, String> requestData, Authentication authentication) {
        String division = requestData.get("division");  // JSON에서 비밀번호 가져오기
        String dtId = authentication.getName();  // 인증된 사용자 ID 가져오기

        try {
            mapper.updateDivision(dtId, division);
            return ResponseEntity.ok(1);  // 성공 시 1 반환
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(0);  // 실패 시 0 반환
        }
    }
}