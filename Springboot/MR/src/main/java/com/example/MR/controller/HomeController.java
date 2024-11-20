package com.example.MR.controller;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;

import com.example.MR.entity.DoctorTb;
import com.example.MR.entity.PatientTb;
import com.example.MR.entity.Person;
import com.example.MR.mapper.MRMapper;

// 화면만 띄우면 되는 곳
@Controller
public class HomeController {
    
    @Autowired
    private MRMapper mapper;
    
    // 초기화면
    @RequestMapping("/index")
    public String index() {
        return "index";
    }
    
	// 메인화면
    @GetMapping("/main")
    public String main() {
    	return "main";
    }
    
    // 서비스소개
    @GetMapping("/about")
    public String about() {
    	return "about";
    }
    
    // 환자 이름 검색 화면
    @GetMapping("/diagnosis")
    public String diagnosis() {
    	return "diagnosis";
    }
    
    // 진단_이미페이지
    @GetMapping("/diagnosis/xray")
    public String xray() {
    	return "xray";
    }
    
    // 회원가입
    @RequestMapping("/signup")
    public String signup() {
    	return "signup";
    }
    // 로그인
    @RequestMapping("/login")
    public String login() {
    	return "login";
    }

    // 마이페이지
    @RequestMapping("/mypage")
    public String mypage() { 
    	return "mypage"; 
    }
    
    
    // 업로드페이지
    @GetMapping("/dicom")
    public String dicom() {
    	return "dicom";
    }
    
    
    
    
    
    // 그냥 테스트
    @GetMapping("/example")
    public String showTable(Model model) {
        List<Person> people = Arrays.asList(
            new Person("Mark", "Otto", "@mdo"),
            new Person("Jacob", "Thornton", "@fat"),
            new Person("Larry", "Bird", "@twitter")
        );
        model.addAttribute("people", people);
        return "test1";
    }
   
    
}