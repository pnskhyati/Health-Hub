import React from 'react';
import Slider from "react-slick";
import "./EPharamaHomeBanner.css";

export default function EPharmaHomeBanner() {
    var settings = {
        dots: true,
        infinite: true,
        speed: 1000,
        slidesToShow: 1,
        slidesToScroll: 1,
        autoplay: true,
        autoplaySpeed: 3000,
        arrows: false,
        centerMode: false,
        swipeToSlide: true,
        edgeFriction: 0.1
    };

    return (
        <>
            <div className="homeBannerSection">
                
                <Slider {...settings}>
                    <div className="item">
                        <img src="HomeBanner1.png" className="img shadow" alt="Banner 1" />
                    </div>
                    <div className="item">
                        <img src="HomeBanner2.png" className="img shadow" alt="Banner 2" />
                    </div>
                    <div className="item">
                        <img src="HomeBanner1.png" className="img shadow" alt="Banner 3" />
                    </div>
                    <div className="item">
                        <img src="HomeBanner2.png" className="img shadow" alt="Banner 4" />
                    </div>
                </Slider>
            </div>
        </>
    );
}
