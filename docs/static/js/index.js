window.HELP_IMPROVE_VIDEOJS = false;

// More Works Dropdown Functionality
function toggleMoreWorks() {
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');

    if (dropdown.classList.contains('show')) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    } else {
        dropdown.classList.add('show');
        button.classList.add('active');
    }
}

// Close dropdown when clicking outside
document.addEventListener('click', function(event) {
    const container = document.querySelector('.more-works-container');
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');

    if (container && !container.contains(event.target)) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Close dropdown on escape key
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        const dropdown = document.getElementById('moreWorksDropdown');
        const button = document.querySelector('.more-works-btn');
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Copy BibTeX to clipboard
function copyBibTeX() {
    const bibtexElement = document.getElementById('bibtex-code');
    const button = document.querySelector('.copy-bibtex-btn');
    const copyText = button.querySelector('.copy-text');

    if (bibtexElement) {
        navigator.clipboard.writeText(bibtexElement.textContent).then(function() {
            // Success feedback
            button.classList.add('copied');
            copyText.textContent = 'Cop';

            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        }).catch(function(err) {
            console.error('Failed to copy: ', err);
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = bibtexElement.textContent;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);

            button.classList.add('copied');
            copyText.textContent = 'Cop';
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        });
    }
}

// Scroll to top functionality
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

// Show/hide scroll to top button
window.addEventListener('scroll', function() {
    const scrollButton = document.querySelector('.scroll-to-top');
    if (window.pageYOffset > 300) {
        scrollButton.classList.add('visible');
    } else {
        scrollButton.classList.remove('visible');
    }

    // Highlight active section in nav menu
    highlightActiveSection();
});

// Highlight active section in navigation
function highlightActiveSection() {
    const sections = document.querySelectorAll('section[id]');
    const navItems = document.querySelectorAll('.nav-item');

    let currentSection = '';
    const scrollPos = window.pageYOffset + 100; // Offset for sticky nav
    const windowHeight = window.innerHeight;
    const documentHeight = Math.max(
        document.body.scrollHeight,
        document.body.offsetHeight,
        document.documentElement.clientHeight,
        document.documentElement.scrollHeight,
        document.documentElement.offsetHeight
    );

    // Check if user is at the bottom of the page or in BibTeX section
    const scrollBottom = window.pageYOffset + windowHeight;
    const isAtBottom = scrollBottom >= documentHeight - 100; // 100px threshold

    // First check if we're in BibTeX section or at bottom
    const bibtexSection = document.getElementById('BibTeX');
    if (bibtexSection) {
        const bibtexTop = bibtexSection.offsetTop;
        const bibtexHeight = bibtexSection.offsetHeight;

        // If we're in BibTeX section or at bottom, highlight BibTeX
        if (scrollPos >= bibtexTop || isAtBottom) {
            currentSection = 'BibTeX';
        }
    }

    // If not in BibTeX, find the current section normally
    if (!currentSection) {
        sections.forEach(section => {
            const sectionId = section.getAttribute('id');
            if (sectionId === 'BibTeX') return; // Skip BibTeX, already checked

            const sectionTop = section.offsetTop;
            const sectionHeight = section.offsetHeight;

            if (scrollPos >= sectionTop && scrollPos < sectionTop + sectionHeight) {
                currentSection = sectionId;
            }
        });
    }

    navItems.forEach(item => {
        item.classList.remove('active');
        const href = item.getAttribute('href');
        if (href === '#' + currentSection) {
            item.classList.add('active');
        }
    });
}

// Add click handlers to nav items to highlight them when clicked
function setupNavClickHandlers() {
    const navItems = document.querySelectorAll('.nav-item');

    navItems.forEach(item => {
        item.addEventListener('click', function(e) {
            const targetId = this.getAttribute('href').substring(1); // Remove the #

            // Remove active class from all items
            navItems.forEach(navItem => navItem.classList.remove('active'));
            // Add active class to clicked item
            this.classList.add('active');

            // After scrolling completes, let the scroll handler take over
            setTimeout(() => {
                highlightActiveSection();
            }, 500);
        });
    });
}

// Video carousel autoplay when in view
function setupVideoCarouselAutoplay() {
    const carouselVideos = document.querySelectorAll('.results-carousel video');

    if (carouselVideos.length === 0) return;

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            const video = entry.target;
            if (entry.isIntersecting) {
                // Video is in view, play it
                video.play().catch(e => {
                    // Autoplay failed, probably due to browser policy
                    console.log('Autoplay prevented:', e);
                });
            } else {
                // Video is out of view, pause it
                video.pause();
            }
        });
    }, {
        threshold: 0.5 // Trigger when 50% of the video is visible
    });

    carouselVideos.forEach(video => {
        observer.observe(video);
    });
}

$(document).ready(function() {
    // Check for click events on the navbar burger icon

    var options = {
		slidesToScroll: 1,
		slidesToShow: 1,
		loop: true,
		infinite: true,
		autoplay: true,
		autoplaySpeed: 5000,
    }

	// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    bulmaSlider.attach();

    // Setup video autoplay for carousel
    setupVideoCarouselAutoplay();

    // Initialize section highlighting
    highlightActiveSection();

    // Setup nav click handlers
    setupNavClickHandlers();

})
