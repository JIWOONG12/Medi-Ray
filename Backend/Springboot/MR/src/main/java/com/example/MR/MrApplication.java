package com.example.MR;

import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cache.annotation.EnableCaching;

@SpringBootApplication
@MapperScan(basePackages = "com.example.MR.mapper")
@EnableCaching
public class MrApplication {

    public static void main(String[] args) {
        SpringApplication.run(MrApplication.class, args);
    }

}