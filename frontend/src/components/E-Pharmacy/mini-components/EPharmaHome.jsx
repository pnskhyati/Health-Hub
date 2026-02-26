import React from 'react'
import EPharmaHomeBanner from './EPharmaHomeBanner'
import LabTests from './LabTests'
import ShopByCategories from './ShopByCategories'
import BestSellers from './BestSellers'

export default function EPharmaHome() {
  
  return (
    <>
     {/* <EPharmaHomeBanner/> */}
     <LabTests/>
     <ShopByCategories/>
     <BestSellers/>
    </>
  )
}
