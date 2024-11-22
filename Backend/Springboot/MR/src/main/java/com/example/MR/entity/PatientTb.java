package com.example.MR.entity;


import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class PatientTb {
	private String ptCode;
	private String ptName;
	private String ptBirthdate;
	private String ptGen; // enum 타입인데 일단 String으로 받기
	private String dtCode;
	private String dtName; // 진단페이지에서 의사코드 대신 의사이름을 뽑기 위해 지정
	private String xrayCode; // xray넘어갈 때 필요한 xrayCode
}
