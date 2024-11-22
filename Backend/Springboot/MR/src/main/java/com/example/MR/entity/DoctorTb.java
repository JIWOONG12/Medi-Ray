package com.example.MR.entity;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor // 모든 필드를 포함하는 생성자 생성
@NoArgsConstructor  // 기본 생성자 생성
@Builder
public class DoctorTb {
    private String DT_CODE;
    private String DT_ID;
    private String DT_PW;
    private String DT_NAME;
    private String DIVISION;
    private String DT_TELNO;
}