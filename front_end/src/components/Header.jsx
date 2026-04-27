import { useState } from 'react';
import { Link } from 'react-router-dom';
import { Activity, Menu, X } from 'lucide-react';

const Header = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const closeMenu = () => {
    setIsMenuOpen(false);
  };

  return (
    <header className="bg-white shadow-md">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <Link to="/" onClick={closeMenu} className="flex items-center space-x-3">
            <Activity className="h-8 w-8 text-primary-600 flex-shrink-0" />

            <div>
              <h1 className="text-xl sm:text-2xl font-bold text-gray-900">
                Fat Predictor
              </h1>
              <p className="text-xs sm:text-sm text-gray-600">
                AI Body Fat Analysis
              </p>
            </div>
          </Link>

          <nav className="hidden md:flex space-x-6">
            <Link to="/" className="text-gray-700 hover:text-primary-600 font-medium">
              Home
            </Link>
            <Link to="/about" className="text-gray-700 hover:text-primary-600 font-medium">
              About
            </Link>
          </nav>

          <button
            type="button"
            onClick={() => setIsMenuOpen((current) => !current)}
            className="md:hidden inline-flex items-center justify-center rounded-lg p-2 text-gray-700 hover:bg-gray-100 hover:text-primary-600 focus:outline-none focus:ring-2 focus:ring-primary-500"
            aria-label={isMenuOpen ? 'Close navigation menu' : 'Open navigation menu'}
            aria-expanded={isMenuOpen}
            aria-controls="mobile-navigation"
          >
            {isMenuOpen ? (
              <X className="h-6 w-6" />
            ) : (
              <Menu className="h-6 w-6" />
            )}
          </button>
        </div>

        {isMenuOpen && (
          <nav
            id="mobile-navigation"
            className="md:hidden mt-4 rounded-xl border border-gray-200 bg-white shadow-sm overflow-hidden"
          >
            <Link
              to="/"
              onClick={closeMenu}
              className="block px-4 py-3 text-gray-700 font-medium hover:bg-gray-50 hover:text-primary-600"
            >
              Home
            </Link>

            <Link
              to="/about"
              onClick={closeMenu}
              className="block px-4 py-3 text-gray-700 font-medium hover:bg-gray-50 hover:text-primary-600 border-t border-gray-100"
            >
              About
            </Link>
          </nav>
        )}
      </div>
    </header>
  );
};

export default Header;