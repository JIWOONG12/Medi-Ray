package com.example.MR.entity;

import java.util.Map;
import lombok.Data;

@Data
public class UploadRequestDto {
	private String image;
    private String ptCode;
    private String fileName;
    private Map<String, Double> modelResult; // JSON의 model_result를 매핑
}
