// Register GSAP plugins first
gsap.registerPlugin(ScrollTrigger);

// Try to init Lenis (non-fatal if CDN didn't load)
let lenis = null;
try {
    lenis = new Lenis({ duration: 1.2, smooth: true });
    lenis.on('scroll', ScrollTrigger.update);
    gsap.ticker.add((time) => { lenis.raf(time * 1000); });
    gsap.ticker.lagSmoothing(0, 0);
    function raf(time) { lenis.raf(time); requestAnimationFrame(raf); }
    requestAnimationFrame(raf);
} catch(e) {
    console.warn('Lenis not loaded, using native scroll.');
}

document.addEventListener("DOMContentLoaded", () => {

    // --- 1. PRELOADER ---
    const chars = document.querySelectorAll('.load-char');
    const percentEl = document.getElementById('loading-percent');

    // Each letter appears one by one
    gsap.to(chars, {
        opacity: 1,
        y: 0,
        duration: 0.4,
        stagger: 0.08,
        ease: 'power3.out'
    });

    // Counter: 0 → 100
    let counter = { val: 0 };
    gsap.to(counter, {
        val: 100,
        duration: 2.5,
        ease: 'power2.inOut',
        onUpdate: () => { percentEl.textContent = Math.round(counter.val); },
        onComplete: () => {
            gsap.to('#preloader', {
                yPercent: -100,
                duration: 1.2,
                ease: 'power4.inOut',
                onComplete: () => {
                    document.getElementById('preloader').style.display = 'none';
                    document.body.classList.remove('loading-locked');
                    animateHero();
                }
            });
        }
    });

    // --- 2. HERO VIDEO CAROUSEL ---
    const heroVideos = document.querySelectorAll('.hero-video');
    let currentIdx = 0;

    setInterval(() => {
        heroVideos[currentIdx].classList.remove('active');
        currentIdx = (currentIdx + 1) % heroVideos.length;
        heroVideos[currentIdx].classList.add('active');
    }, 6000);

    // --- 3. HERO TEXT ENTRANCE ---
    function animateHero() {
        gsap.fromTo('.top-left .giant-text',
            { y: 40, opacity: 0 },
            { y: 0, opacity: 1, duration: 1.2, ease: 'power3.out' }
        );
        gsap.fromTo('.top-left .sub-label',
            { y: 20, opacity: 0 },
            { y: 0, opacity: 1, duration: 1, delay: 0.3, ease: 'power3.out' }
        );
        gsap.fromTo('.bottom-right .lede',
            { y: 30, opacity: 0 },
            { y: 0, opacity: 1, duration: 1, delay: 0.5, ease: 'power3.out' }
        );
        gsap.fromTo('.bottom-right .sub-text',
            { y: 30, opacity: 0 },
            { y: 0, opacity: 1, duration: 1, delay: 0.7, ease: 'power3.out' }
        );
    }

    // --- 4. DROPDOWN MENU (toggle with class) ---
    const menuBtn = document.getElementById('menu-btn');
    const menuOverlay = document.getElementById('fullscreen-menu');
    const menuLinks = document.querySelectorAll('.menu-link');
    let isMenuOpen = false;

    menuBtn.addEventListener('click', () => {
        isMenuOpen = !isMenuOpen;
        menuOverlay.classList.toggle('open', isMenuOpen);
        menuBtn.textContent = isMenuOpen ? 'Close' : 'Menu';
    });

    menuLinks.forEach(link => {
        link.addEventListener('click', () => {
            isMenuOpen = false;
            menuOverlay.classList.remove('open');
            menuBtn.textContent = 'Menu';
        });
    });

    // --- 5. NAVBAR: HIDE ON SCROLL DOWN, SHOW ON SCROLL UP ---
    const navBar = document.querySelector('.nav-bar');
    let lastScroll = 0;

    window.addEventListener('scroll', () => {
        const currentScroll = window.scrollY;
        if (currentScroll <= 50) {
            // Always show at the very top
            navBar.classList.remove('hidden');
        } else if (currentScroll > lastScroll) {
            // Scrolling DOWN → hide
            navBar.classList.add('hidden');
            // Also close menu if open
            if (isMenuOpen) {
                isMenuOpen = false;
                menuOverlay.classList.remove('open');
                menuBtn.textContent = 'Menu';
            }
        } else {
            // Scrolling UP → show
            navBar.classList.remove('hidden');
        }
        lastScroll = currentScroll;
    });

    // --- 6. SCROLL ANIMATIONS ---

    // Fade-in-up elements
    gsap.utils.toArray('.fade-in-up').forEach(el => {
        gsap.fromTo(el,
            { y: 40, opacity: 0 },
            { y: 0, opacity: 1, duration: 1, ease: 'power3.out',
              scrollTrigger: { trigger: el, start: 'top 88%' }
            }
        );
    });

    // Story text slides in from the left
    gsap.utils.toArray('.story-text').forEach(text => {
        gsap.fromTo(text,
            { x: -60, opacity: 0 },
            { x: 0, opacity: 1, duration: 1.2, ease: 'power3.out',
              scrollTrigger: { trigger: text, start: 'top 85%' }
            }
        );
    });

    // Image reveal with mask wipe + subtle parallax
    gsap.utils.toArray('.img-reveal').forEach(container => {
        const mask = container.querySelector('.img-mask');
        const img = container.querySelector('img');

        gsap.set(mask, { clipPath: 'inset(100% 0 0 0)' });
        gsap.set(img, { scale: 1.3 });

        // Clip-path wipe reveal
        gsap.to(mask, {
            clipPath: 'inset(0% 0 0 0)',
            duration: 1.4,
            ease: 'power4.out',
            scrollTrigger: { trigger: container, start: 'top 80%' }
        });

        // Slow parallax zoom-out while scrolling past
        gsap.to(img, {
            scale: 1,
            ease: 'none',
            scrollTrigger: {
                trigger: container,
                start: 'top bottom',
                end: 'bottom top',
                scrub: 1
            }
        });
    });

    // Stagger feature items
    gsap.utils.toArray('.feature-item').forEach((item, i) => {
        gsap.fromTo(item,
            { x: -30, opacity: 0 },
            { x: 0, opacity: 1, duration: 0.8, delay: i * 0.15, ease: 'power3.out',
              scrollTrigger: { trigger: item, start: 'top 90%' }
            }
        );
    });

    // Dashboard section scales up gently
    gsap.fromTo('.dashboard-container',
        { y: 60, opacity: 0, scale: 0.97 },
        { y: 0, opacity: 1, scale: 1, duration: 1.2, ease: 'power3.out',
          scrollTrigger: { trigger: '.dashboard-container', start: 'top 85%' }
        }
    );

    // --- 6. DASHBOARD LOGIC ---
    let currentModel = 'vit', currentView = 'map';

    const modelDesc = {
        vit: '<strong>Vision Transformer (Baseline):</strong> Splits surface maps into patches, encoding them with a lightweight Transformer. Reconstructs subsurface temperatures from purely surface inputs.',
        lstm: '<strong>ConvLSTM Lag Network:</strong> Leverages temporal dynamics via a 3-day memory lag. Incorporates prior states to produce highly accurate, time-dependent fluid dynamic forecasts.'
    };
    const modelNames = { vit: 'Vision Transformer', lstm: 'ConvLSTM' };
    const viewNames = { map: 'Prediction Map', profile: 'Vertical Profile', metrics: 'Validation Metrics' };

    const viewPanels = document.querySelectorAll('.view-panel');
    const viewerTitle = document.getElementById('viewer-title');
    const descEl = document.getElementById('model-desc');

    function updateViewer() {
        viewerTitle.textContent = `${modelNames[currentModel]} — ${viewNames[currentView]}`;
        descEl.innerHTML = modelDesc[currentModel];
        viewPanels.forEach(p => p.classList.remove('active'));
        const target = document.getElementById(`view-${currentModel}-${currentView}`);
        if (target) {
            target.classList.add('active');
            gsap.fromTo(target, { opacity: 0, scale: 0.98 }, { opacity: 1, scale: 1, duration: 0.4, ease: 'power2.out' });
        }
    }

    document.querySelectorAll('.model-tabs .tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.model-tabs .tab-btn').forEach(t => t.classList.remove('active'));
            btn.classList.add('active');
            currentModel = btn.dataset.model;
            updateViewer();
        });
    });

    document.querySelectorAll('.output-tabs .tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.output-tabs .tab-btn').forEach(t => t.classList.remove('active'));
            btn.classList.add('active');
            currentView = btn.dataset.view;
            updateViewer();
        });
    });

});
