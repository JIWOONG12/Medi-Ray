package com.example.MR.entity;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class ImgTb {
	private String xrayCode;
	private String ptCode;
	private String xrayDate;
	private String xrayImgPath;
	private String dtCode;
	private String dtOpinion;
	
}
