package com.example.MR.service;


import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.userdetails.User;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;

import com.example.MR.entity.DoctorTb;
import com.example.MR.mapper.MRMapper;

@Service
public class CustomUserDetailsService implements UserDetailsService {

	@Autowired
    private final MRMapper mapper;

    public CustomUserDetailsService(MRMapper mapper) {
        this.mapper = mapper;
    }

    @Override
    public UserDetails loadUserByUsername(String dtId) throws UsernameNotFoundException {
        // 데이터베이스에서 사용자 조회
    	System.out.println("test");
    	if (dtId == null) {
            throw new UsernameNotFoundException("사용자 이름이 비어 있습니다.");
        }
        DoctorTb doctor = mapper.checkDuplicateId(dtId); // username을 기준으로 사용자 조회
        
        
        
        if (doctor == null) {
            throw new UsernameNotFoundException("존재하는 ID가 없습니다.: " + dtId);
        }
        
        System.out.println("ID: " + doctor.getDT_ID());
        System.out.println("Password: " + doctor.getDT_PW());
        

        // UserDetails 객체로 반환
        UserDetails user = User.builder()
                .username(doctor.getDT_ID()) // 사용자 ID 설정
                .password(doctor.getDT_PW()) // 사용자 암호화된 비밀번호 설정
//                .roles(doctor.getDtRole()) // 사용자 역할 설정
                .build();
        
        System.out.println("UserDetails: " + user); // UserDetails 객체 확인

        return user;
    }
}
