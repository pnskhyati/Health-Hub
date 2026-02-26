import React, { useEffect, useState } from 'react';
import { Grid, Card, CardContent, Typography, Button, Select, MenuItem, FormControl, InputLabel } from '@mui/material';
import { useCart } from '../Cart/CartContext';
import './ProductDetails.css';
import axios from 'axios';
import { FaStar, FaStarHalfAlt, FaRegStar } from 'react-icons/fa';
import ShoppingHome from "../mini-components/ShoppingHome";
import SingleProduct from '../Products/SingleProduct';

export default function ProductDetails() {
  const { addToCart } = useCart();
  const [products, setProducts] = useState([]);
  const [selectedCategory, setSelectedCategory] = useState('');
  const [sortPrice, setSortPrice] = useState('');
  const categories = ['Pain Relief', 'Cold & Cough', 'Vitamins'];
  const [openDialog, setOpenDialog] = useState(false);
  const [selectedProduct, setSelectedProduct] = useState(null);

  const openProductDialog = (product) => {
    setSelectedProduct(product); // Set the selected product details
    setOpenDialog(true);        // Open the dialog
  };

  const closeDialog = () => {
    setOpenDialog(false);       // Close the dialog
    setSelectedProduct(null);   // Clear the selected product
  };

  useEffect(() => {
    async function fetchProducts() {
      try {
        const response = await axios.get("http://localhost:5000/medicines");
        setProducts(response.data);
        console.log("response: ", response.data);
      } catch (error) {
        console.log('Error fetching products: ', error);
      }
    }
    fetchProducts();
  }, []);

  const handleAddToCart = (product) => {
    console.log("product clicked:", product);
    addToCart(product);
  };

  const handleCategoryChange = (event) => {
    setSelectedCategory(event.target.value);
  };

  const handleSortPriceChange = (event) => {
    setSortPrice(event.target.value);
  };

  const filteredProducts = products.filter(product =>
    selectedCategory ? product.category === selectedCategory : true
  );

  const sortedProducts = filteredProducts.sort((a, b) => {
    if (sortPrice === 'asc') return parseFloat(a.MRP.replace('₹', '')) - parseFloat(b.MRP.replace('₹', ''));
    if (sortPrice === 'desc') return parseFloat(b.MRP.replace('₹', '')) - parseFloat(a.MRP.replace('₹', ''));
    return 0;
  });

  const renderStars = (rating) => {
    const validRating = Number(rating);

    if (isNaN(validRating) || validRating < 0 || validRating > 5) {
      return null;
    }

    const fullStars = Math.floor(validRating);
    const halfStars = validRating % 1 !== 0;
    const emptyStars = 5 - fullStars - (halfStars ? 1 : 0);

    return (
      <>
        {[...Array(fullStars)].map((_, i) => <FaStar key={`full-${i}`} />)}
        {halfStars && <FaStarHalfAlt />}
        {[...Array(emptyStars)].map((_, i) => <FaRegStar key={`empty-${i}`} />)}
      </>
    );
  };

  return (
    <>
      <ShoppingHome />
      <div className="product-details-page">
        <div className="sidebar">
          <h3>Filter & Sort</h3>
          <FormControl fullWidth>
            <InputLabel id="category-select-label">Category</InputLabel>
            <Select
              labelId="category-select-label"
              value={selectedCategory}
              label="Category"
              onChange={handleCategoryChange}
            >
              <MenuItem value="">All Categories</MenuItem>
              {categories.map((category, index) => (
                <MenuItem key={index} value={category}>{category}</MenuItem>
              ))}
            </Select>
          </FormControl>

          <FormControl fullWidth>
            <InputLabel id="price-select-label">Sort by Price</InputLabel>
            <Select
              labelId="price-select-label"
              value={sortPrice}
              label="Sort by Price"
              onChange={handleSortPriceChange}
            >
              <MenuItem value="">None</MenuItem>
              <MenuItem value="asc">Low to High</MenuItem>
              <MenuItem value="desc">High to Low</MenuItem>
            </Select>
          </FormControl>
        </div>

        <div className="products-list">
          <Grid container spacing={3}>
            {sortedProducts.map((product) => (
              <Grid item xs={12} sm={6} md={4} key={product._id}>
                <Card className="product-card">
                  <img src="https://www.netmeds.com/images/product-v1/150x150/908000/sebamed_baby_protective_facial_cream_100ml_121408_0_1.jpg" alt={product.MedicineName} className="product-image" />
                  <CardContent>
                    <Typography variant="h6" onClick={() => openProductDialog(product)}>{product.MedicineName}</Typography>
                    <Typography variant="body2" color="textSecondary">{product.Manufacturer}</Typography>
                    <div className="product-rating">
                      {renderStars(product.Ratings)}
                    </div>
                    <div className="price-section">
                      <Typography variant="body2" color="textSecondary" className="striked-price">₹ {product.MRP}</Typography>
                      <Typography variant="body1" color="primary" className="product-price">₹ {product.bestPrice}</Typography>
                      <Typography variant="body2" color="green">
                      ({Math.round(((product.MRP - product.bestPrice) / product.MRP) * 100)}% OFF)
                    </Typography>
                    </div>
                    <Button
                      variant="contained"
                      color="secondary"
                      onClick={() => handleAddToCart(product)}
                      className="add-to-cart-button"
                    >
                      Add to Cart
                    </Button>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </div>

        {/* SingleProduct Dialog */}
        {selectedProduct && (
          <SingleProduct
            product={selectedProduct}
            openDialog={openDialog}
            closeDialog={closeDialog}
          />
        )}
      </div>
    </>
  );
}
