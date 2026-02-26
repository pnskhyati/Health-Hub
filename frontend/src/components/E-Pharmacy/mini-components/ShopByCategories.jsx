import React from 'react';
import './ShopByCategories.css';

const categories = [
  { id: 1, name: 'Fitness', image: 'categories3.png' },
  { id: 2, name: 'Personal Care', image: 'categories4.png' },
  { id: 3, name: 'Healthcare Devices', image: 'categories5.png' },
  { id: 4, name: 'Home Care', image: 'categories2.png' },
  { id: 5, name: 'Mother and Baby Care', image: 'categories1.png' },

];

export default function ShopByCategories() {
  return (
    <div className="categories-container">
      <h2 className="categories-title">Shop by Categories</h2>
      <div className="categories-grid">
        {categories.map(category => (
          <div className="category-box" key={category.id}>
            <img src={category.image} alt={category.name} className="category-image" />
            <div className="category-overlay">
              <p className="category-name">{category.name}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
